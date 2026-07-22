from typing import Dict, Any, List, Callable
from .search import tool_search, tool_deep_research, tool_fact_check
from .darkweb_search import tool_darkweb_search, tool_darkweb_research
from .database import tool_postgres_admin
from .file_system import tool_file_system
from .tasks import tool_manage_tasks
from .system import tool_system_utility
from .memory import tool_knowledge_base, tool_recall, tool_unified_forget, tool_update_profile, tool_learn_skill, tool_scratchpad
from .execute import tool_execute
from .browser import tool_browser
from .sandbox_services import (
    tool_manage_services, MANAGE_SERVICES_TOOL_DEFINITION,
)
from .delegate import (
    tool_delegate, tool_jobs,
    DELEGATE_TOOL_DEFINITION, JOBS_TOOL_DEFINITION,
)
from .notify_tool import (
    tool_notify_operator, NOTIFY_OPERATOR_TOOL_DEFINITION,
)
from .swarm import tool_delegate_to_swarm
from .acquired_skills import tool_create_skill, tool_manage_skills, AcquiredSkillManager
from .composed_skills import (
    register_composed_skills,
    register_composed_skill_runners,
    tool_manage_composed_skills,
)
from .projects import tool_manage_projects, MANAGE_PROJECTS_TOOL_DEF
from .self_state import tool_self_state
from .introspect import tool_introspect
from .postmortem_review import tool_postmortem
from .workspace import tool_workspace
from .workspace_track import tool_workspace_track
from .uncertainty_tool import tool_flag_uncertainty

import logging
import re
from ..utils.logging import pretty_log, Icons


def _acquired_skill_result_ok(result) -> bool:
    """Classify an acquired-skill execution result as success/failure.

    `tool_execute` RETURNS error strings (non-zero EXIT CODE, tracebacks,
    [SYSTEM ERROR]) rather than raising, so an unconditional success=True
    telemetry write recorded broken skills as wins — resetting
    failure_count and defeating degraded-skill retirement.
    """
    s = str(result)
    if "[SYSTEM ERROR]" in s or "Critical Tool Error" in s:
        return False
    m = re.search(r"EXIT CODE:\s*(\d+)", s)
    if m:
        return m.group(1) == "0"
    # No exit-code banner (non-execute-shaped result): success unless it
    # clearly starts with an error marker.
    return not s.lstrip().startswith(("Error", "ERROR", "SYSTEM ERROR", "Traceback"))

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
            "description": "Unified file manager. ALWAYS use this to list, read, write. Use operation='search' for instantaneous high-performance ripgrep text searching across the codebase. Use operation='find' to locate files by wildcard name (e.g. '*.py'). LARGE FILES: do NOT emit a huge file (more than ~500 lines / ~40KB) in a single 'write' — a tool call that big frequently exceeds the JSON arg limit and fails to parse (the content is then lost). Instead write a compact SKELETON first, then grow it with successive operation='replace' edits that insert one section at a time. Build big single-file apps incrementally, not in one mega-write.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "read_chunked", "inspect", "search", "find", "list_files", "write", "replace", "download", "copy", "rename", "move", "delete"],
                        "description": "The exact operation to perform. Use 'write' to create a new file OR completely overwrite an existing one (provide the FULL file in 'content'). Use 'replace' for TARGETED edits to a small region of an existing file — ALWAYS PREFER the single-argument form: put ONE block in 'content' shaped exactly like `<<<< SEARCH\\n<exact current text>\\n====\\n<new text>\\n>>>>` and OMIT 'replace_with' entirely (this form survives argument-transport corruption; concatenate several blocks for multiple edits in one call). The legacy two-argument form (content=old block + replace_with=new block) still works but is fragile in transport. If you are rewriting the whole file, always use 'write', not 'replace'."
                    },
                    "path": {
                        "type": "string",
                        "description": "The target file or directory path relative to the active project root. For operation='list_files', pass a subdirectory to list just that subtree (omit for the workspace root). Container-absolute '/workspace/...' paths (as printed by `execute` shell output) are also accepted and resolve to the same files the shell sees."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional for operation='read': 1-based first line of a LINE-RANGE read. Returns only that slice with line-number prefixes and is EXEMPT from the whole-file size cap — the cheap way to re-read one region after a failed 'replace' or a too-large-file error. Chains directly from 'search' line numbers. Pair with end_line."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional for operation='read': 1-based last line of the range (inclusive). Omit to read from start_line to the end (capped)."
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
                        "description": "MANDATORY for 'write': the FULL new file contents. MANDATORY for 'replace': PREFERRED — an Aider-style block `<<<< SEARCH\\n<exact current text>\\n====\\n<new text>\\n>>>>` with 'replace_with' omitted (immune to argument-transport corruption); LEGACY — the exact EXISTING code block to find, paired with 'replace_with'. If you want to rewrite the whole file, use operation='write', not 'replace'."
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
                        "description": "For the LEGACY two-argument 'replace' form only: the new code/text that takes the place of the existing block in 'content'. PREFER omitting this entirely and putting Aider-style `<<<< SEARCH ==== >>>>` block(s) inside 'content' — the old→new pairs then live in ONE argument, which survives argument-transport corruption. NEVER send the same text in 'content' and 'replace_with'; such a call is rejected. If you meant to rewrite the whole file, use operation='write' instead."
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
    {"type": "function", "function": {"name": "knowledge_base", "description": "Memory manager: imports EXISTING files (ingest_document), ASKS A QUESTION AGAINST ONE INGESTED DOCUMENT (query), records discrete facts (insert_fact), forgets a topic, lists or wipes the store. NEVER use to compose, draft, or save prose the user just asked you to write — answer the user directly instead. Do NOT write Python scripts to read PDFs or ingest files.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["insert_fact", "ingest_document", "query", "expand", "forget", "list_docs", "reset_all"]}, "filename": {"type": "string", "description": "REQUIRED for ingest_document (an EXISTING local filename in the sandbox, e.g. 'plan.txt', or a web HTML URL — NOT raw prose, NOT a title, NOT a PDF URL). REQUIRED for query (which ingested document to ask). REQUIRED for forget (the topic name to forget)."}, "question": {"type": "string", "description": "REQUIRED for action='query': the question to answer from that ONE document. Returns the best-matching passages WITH their section breadcrumbs. This is the RIGHT tool for a large reference manual (e.g. an ingested PostgreSQL PDF) — far better than `recall`, which searches ALL memory at once and caps at a shared budget. Call it repeatedly with different wording to drill down."}, "fact": {"type": "string", "description": "REQUIRED for insert_fact only: a single discrete fact to memorize (e.g. 'The user lives in Athens'). Do NOT use for ingest_document — pass a filename via 'filename' instead."}, "ref": {"type": "string", "description": "REQUIRED for action='expand': an evidence handle from a recall hit's EVIDENCE REFS line (e.g. 'ep:12' for the full episode record, 'session:<id>' for a stored conversation). Expands an abstraction back to its raw source."}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "recall", "description": "Search the vector database for facts and answers from past conversations and general memory. ALSO the right tool when a question names a project or topic that manage_projects doesn't track ('when does project X ship?') — facts about it often live here even when no tracked project exists. For a question about ONE specific INGESTED DOCUMENT (e.g. a manual or PDF), prefer knowledge_base(action='query', filename=..., question=...) — it searches only that document and returns more, better-scoped passages. Hits may carry EVIDENCE REFS (e.g. 'ep:12') — expand one via knowledge_base(action='expand', ref=...). WARNING: This retrieves semantic chunks; use file_system operation='search' if you need an exact line match.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The specific question or search query."}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "execute", "description": "Run Python, Node.js, or Shell code. USE THIS ONLY AS A LAST RESORT for custom math, logic, or formatting. DO NOT use this to simply create/write web files (HTML/CSS) or data files (use file_system write instead!). DO NOT use this to download files, scrape the web, or manage memory. WARNING: Native tools CANNOT be imported in Python. ALWAYS print results. To run an already existing file, you can omit the 'content' parameter. The sandbox has its OWN pid namespace and loopback: you CANNOT kill, restart, or reach a process the USER runs on their machine — if they run it, ask them to restart it.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Optional. A direct bash command to run immediately (e.g., 'ls -la' or 'cat file.csv'). If provided, filename and content are ignored."}, "filename": {"type": "string", "description": "Optional. The name of the file to execute. If omitted but 'content' is provided, an ephemeral script will be generated and run automatically."}, "content": {"type": "string", "description": "The code to execute. Omit this if you just want to run an existing file."}, "args": {"type": "array", "items": {"type": "string"}, "description": "Optional command line arguments to safely pass to the script."}, "stateful": {"type": "boolean", "description": "If true, Python variables/dataframes/models are saved and automatically loaded into memory for your next execution. Acts like a Jupyter Notebook cell."}}, "required": []}}},
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
                        "description": "navigate: open URL, return status/title AND a capped ~8KB `text` preview of the page — you usually do NOT need a follow-up extract_text. extract_text: return the FULL body innerText (or a specific selector) up to max_chars — use only when the navigate preview was truncated or you need a precise selector. click: click a CSS selector, returns url/title AND the post-click `text` preview. screenshot: save PNG to /workspace. close: clear the persistent session profile (and the last-URL sidecar). interact: run a sequence of sub-actions (in `actions`) in ONE Chromium context — required for multi-step SPA flows where the atomic per-op re-navigation would wipe intermediate DOM state."
                    },
                    "url": {"type": "string", "description": "Target URL. REQUIRED for 'navigate'. OPTIONAL for extract_text / click / screenshot / interact: when omitted, the tool re-navigates to the URL of the most recent navigate/extract/click/screenshot/interact call (stored in <profile_dir>/.last_url). If neither an explicit URL nor a sidecar URL is available, the tool errors with a clear 'call navigate first or pass url=...' message."},
                    "selector": {"type": "string", "description": "CSS selector. Required for 'click'. Optional for 'extract_text' to narrow to a specific element."},
                    "out_path": {"type": "string", "description": "PNG output path for 'screenshot'. Relative to /workspace. Defaults to 'screenshot.png'."},
                    "actions": {
                        "type": "array",
                        "description": "REQUIRED for operation='interact'. Ordered list of sub-actions executed inside a single Chromium context. Each item is one dict with an `action` field and action-specific params. Failures are reported per-action (the sequence keeps going by default; set stop_on_error=true to short-circuit). Supported actions:\n  {\"action\":\"goto\",\"url\":\"...\",\"wait_until\":\"load\"}\n  {\"action\":\"click\",\"selector\":\"...\",\"wait_for_hidden\":\"#overlay\",\"force\":false}\n  {\"action\":\"dblclick\",\"selector\":\"...\"}   # use when the element's open/launch handler is bound to ondblclick (desktop-icon-style UIs). A plain `click` will NOT fire dblclick listeners.\n  {\"action\":\"extract_text\",\"selector\":\"...\",\"max_chars\":65536}\n  {\"action\":\"fill\",\"selector\":\"...\",\"text\":\"...\"}\n  {\"action\":\"wait_for_selector\",\"selector\":\"...\",\"state\":\"visible|hidden|attached|detached\",\"timeout_ms\":5000}\n  {\"action\":\"screenshot\",\"out_path\":\"...\"}\n  {\"action\":\"sleep\",\"ms\":500}\nclick / dblclick / fill accept TWO optional fields beyond `selector`:\n  - `wait_for_hidden`: a CSS selector for an OVERLAY that must disappear before the click fires (e.g. a fading lock-screen, modal backdrop, or splash). The action waits for that selector to leave the visible state, then proceeds. If it doesn't disappear in time, the action fails with a clear message — better than Playwright's generic 'element intercepts pointer events'.\n  - `force`: skip Playwright's actionability check (visibility/stability/hit-test). Use sparingly when you're certain the target is correct but Playwright is being conservative (e.g. an element mid-CSS-transition).\nwait_for_selector accepts `state` ∈ {visible (default), hidden, attached, detached}. Use `state=hidden` to wait for an overlay to GO AWAY — bare wait_for_selector waits for an element to APPEAR, which is useless for fade-out flows.\nExample — open a desktop-icon-based app in a WebOS, waiting for the fading lock screen to clear first:\n  actions=[\n    {\"action\":\"click\",\"selector\":\"#unlock-btn\"},\n    {\"action\":\"wait_for_selector\",\"selector\":\"#lock-screen\",\"state\":\"hidden\"},\n    {\"action\":\"dblclick\",\"selector\":\".desktop-icon[data-app='calculator']\"},\n    {\"action\":\"wait_for_selector\",\"selector\":\"#calc-display\"},\n    {\"action\":\"screenshot\",\"out_path\":\"calc_opened.png\"}\n  ]",
                        "items": {"type": "object"}
                    },
                    "stop_on_error": {"type": "boolean", "description": "interact only. If true, the sequence aborts on the first failing action; otherwise (default) it continues and reports all failures. Handy for probing which sub-action is flaky."},
                    "wait_until": {"type": "string", "enum": ["load", "domcontentloaded", "networkidle"], "description": "Navigation readiness threshold. Default 'load'. Use 'domcontentloaded' for slow pages that never idle, 'networkidle' for SPAs."},
                    "full_page": {"type": "boolean", "description": "Screenshot: capture entire scrollable page (default true) vs. just the viewport."},
                    "max_chars": {"type": "integer", "description": "extract_text: cap returned text at this many chars (default 65536)."},
                    "timeout_ms": {"type": "integer", "description": "Per-operation (or per-action, for interact) timeout in ms. Default 30000. The overall interact budget is (timeout_ms × len(actions)), so large sequences get proportionally more wall time."},
                    "settle_ms": {"type": "integer", "description": "screenshot: wait this many ms after page load before capturing — lets a WebGL/canvas scene paint its first frames instead of shooting a blank."},
                    "click_center": {"type": "boolean", "description": "screenshot: click the viewport centre before capturing (then wait post_click_ms, default 800). Use for pointer-lock/canvas games that only start rendering after a click — captures what a USER would actually see."},
                    "nav_text_chars": {"type": "integer", "description": "navigate/click: size of the inline page-text preview in chars (default 8192, 0 disables, capped at 65536)."}
                },
                "required": ["operation"]
            }
        }
    },
    MANAGE_SERVICES_TOOL_DEFINITION,
    DELEGATE_TOOL_DEFINITION,
    JOBS_TOOL_DEFINITION,
    NOTIFY_OPERATOR_TOOL_DEFINITION,
    {"type": "function", "function": {"name": "learn_skill", "description": "MANDATORY when you solve a complex bug or task after initial failure. Save the lesson so you don't repeat the mistake.", "parameters": {"type": "object", "properties": {"task": {"type": "string"}, "mistake": {"type": "string"}, "solution": {"type": "string"}}, "required": ["task", "mistake", "solution"]}}},
    {"type": "function", "function": {"name": "flag_uncertainty", "description": "Register what you DON'T know or are unsure about with your metacognitive tracker. Call action='unknown' when you need a fact you don't have (set impact 1-5 — 4+ means it materially affects correctness; resolution tells how to get it: 'ask user', 'search web', 'read file'). Call action='assumption' when you are proceeding on a belief you have NOT verified (set confidence 0.0-1.0). action='list' shows what is currently flagged plus recurring blind-spots from past turns. A critical unknown (impact>=4, resolution='ask user') triggers a clarification prompt before your answer is finalized — so flag honestly rather than guessing. Everything flagged persists, so questions you keep hitting become visible as recurring blind-spots.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["unknown", "assumption", "list"]}, "text": {"type": "string", "description": "For action='unknown': what you don't know. For action='assumption': the unverified belief."}, "impact": {"type": "integer", "minimum": 1, "maximum": 5, "description": "For action='unknown': how much not knowing this affects correctness (1 minor, 5 critical)."}, "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "For action='assumption': how confident you are in the belief (0.0-1.0)."}, "resolution": {"type": "string", "description": "For action='unknown': how to resolve it — 'ask user', 'search web', 'read file', etc."}, "basis": {"type": "string", "description": "For action='assumption': why you believe it."}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "workspace", "description": "READ-ONLY view of the user's WORKSPACE state — what's outside of you (files, scheduled-task outcomes, research artifacts you've pulled, commands you ran). This is the world-model counterpart to introspect (which reads your selfhood). Use this when the user asks 'what changed since yesterday?', 'what did my scheduled task do?', 'have I already pulled this URL?', 'show me what you've been doing in my project'. Distinct from: introspect (your own selfhood), file_system (one-shot reads of the filesystem), recall (vector search over ingested docs). Actions: 'summary' (default; stats + narrative + recent changes + recent tasks/research); 'stats' (counts); 'files' (the watchlist); 'changes' (diff tracked files against last-seen snapshot); 'tasks' (recent scheduled-task outcomes); 'research' (URLs you've already pulled); 'commands' (significant command outcomes); 'narrative' (the running workspace summary); 'recent' (the activity log, mixed kinds); 'search' (keyword search over the activity log — pass 'query').", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["summary", "stats", "files", "changes", "tasks", "research", "commands", "narrative", "recent", "search"], "description": "Default 'summary' if omitted."}, "limit": {"type": "integer", "description": "For tasks/research/commands/recent/search: how many entries to return. Defaults to 10; capped at 50."}, "query": {"type": "string", "description": "For action='search': keywords to find in past workspace events (filenames, commands, URLs, task names)."}}, "required": []}}},
    {"type": "function", "function": {"name": "workspace_track", "description": "WRITE path into the WORKSPACE state — author the watchlist of files to track, free-form workspace notes, and manual research dedup markers. Counterpart to the read-only workspace tool. Actions: 'track' (add a file path to the watchlist; optional 'label' for a human descriptor); 'untrack' (remove a path); 'note' (record a free-form workspace observation); 'mark_seen' (record a URL as already-pulled so future research dedups against it). Tracked files get a stat-cache diff on every wake-up, so 'track' is how you get 'what changed in this file since last session' to surface automatically.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["track", "untrack", "note", "mark_seen"]}, "path": {"type": "string", "description": "For track/untrack: the file path to add/remove from the watchlist."}, "label": {"type": "string", "description": "Optional for track: a short human descriptor (e.g. 'main config', 'experiment log')."}, "text": {"type": "string", "description": "Required for note: the free-form observation."}, "url": {"type": "string", "description": "Required for mark_seen: the URL to record as already-pulled."}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "introspect", "description": "READ-ONLY introspection over your OWN selfhood — your running first-person diary, recent experiences, topic clusters, and counts. Use this when the user asks you to describe yourself, what you've been working on, what you remember, or what you've done before. Distinct from: self_state (which AUTHORS open questions / threads / mood for the next session), knowledge_base (facts about the world), update_profile (facts about the USER). Actions: 'summary' (default; renders stats + running diary + recent experiences in one block — the natural answer to 'tell me about yourself'); 'stats' (counts and the topic cluster mix); 'narrative' (just the running diary); 'recent' (the last N first-person experiences); 'recall' (relevance-ranked search over your past, IDF-weighted, no embeddings — pass 'query'); 'activity' (the background-activity ledger: dream/REM cycles, PRM/router/calibration retrains, skills graduated, self-play, scheduled-task conclusions — THE answer to 'what did you do while I was away?' / 'what ran in the background?', since routine maintenance no longer auto-surfaces in replies; optional 'hours' window and 'limit'). All reads route through your SelfModel; nothing here writes.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["summary", "stats", "narrative", "recent", "recall", "activity"], "description": "Default 'summary' if omitted."}, "query": {"type": "string", "description": "Required for action='recall': what to search your past for (e.g. 'postgres migrations', 'the trapdoor question')."}, "limit": {"type": "integer", "description": "For action='recent'/'recall': how many results (default 5, cap 25). For action='activity': how many ledger lines (default 30, cap 100)."}, "hours": {"type": "number", "description": "For action='activity': look-back window in hours. Default 24; capped at 336 (14 days)."}}, "required": []}}},
    {"type": "function", "function": {"name": "postmortem", "description": "READ-ONLY view of your post-mortem DEFECT QUEUE — the durable, classified findings your idle-time post-mortem engine files after analysing the whole transcript of your worst FAILED runs. Use this when asked 'what have you found broken in yourself?', 'what defects are open?', 'show me the post-mortem of that bad run', or to review a proposed fix before it's applied by a human. Each defect is one of: 'behavioural' (you chose badly — already routed to a lesson), 'configuration' (a flag/threshold let it through), or 'code_defect' (a tool/loop is broken — may carry a proposed reproducing test + diff, stored for review, NEVER auto-applied). Distinct from: introspect (your selfhood/diary), workspace (the user's world). Actions: 'pending' (default; open defects, worst first); 'list' (all, any status); 'show' (full detail incl. any proposed test/patch — pass 'defect_id'); 'stats' (counts by category/status).", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["pending", "list", "show", "stats"], "description": "Default 'pending' if omitted."}, "defect_id": {"type": "string", "description": "Required for action='show': the defect id (or an id-prefix) from a 'pending'/'list' result."}, "limit": {"type": "integer", "description": "For 'pending'/'list': how many to return. Defaults to 10; capped at 25."}}, "required": []}}},
    {"type": "function", "function": {"name": "self_state", "description": "Author your OWN cross-session continuity state — the open questions, unfinished threads, and mood you carry from this session into the next. This is YOUR forward-looking self, not facts about the world. Use it when you finish a turn but something is left unresolved that the next session of you should pick up. Distinct from: knowledge_base (facts/documents), update_profile (facts about the USER), scratchpad (notes for THIS conversation only). action='note_question' records something you are still trying to figure out; 'resolve_question' marks one answered; 'add_unfinished' notes a task left mid-flight; 'close_unfinished' completes one; 'set_mood' records your current functional state (e.g. 'curious', 'stuck', 'satisfied'); 'note_principle' records an operating principle — how you CHOOSE to work (e.g. 'I verify before asserting', 'I prefer reversible actions') — surfaced in your wake-up prefix every session to shape your behaviour; 'list' shows what is currently on file. Whatever you record here is shown to you at the start of your next session.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["note_question", "resolve_question", "add_unfinished", "close_unfinished", "set_mood", "note_principle", "list"]}, "text": {"type": "string", "description": "For note_question/add_unfinished/note_principle: the question, thread, or principle text. For resolve_question/close_unfinished: the id (or id-prefix, or a text substring) of the item to close."}, "mood": {"type": "string", "description": "Required for set_mood: a short functional-state label (e.g. 'curious', 'stuck', 'satisfied')."}, "evidence": {"type": "string", "description": "Optional for set_mood: one sentence on why."}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the internet (Anonymous via Tor). ALWAYS use this FIRST for simple factual questions and general web searches. CRITICAL: Keep your queries concise and keyword-focused (e.g., 'PostgreSQL 16 release notes'). DO NOT use long conversational sentences. PLAIN KEYWORDS ONLY — do NOT use search operators like 'site:', quoted \"exact phrases\", or boolean OR/AND. The search runs over Tor against scraper backends (DuckDuckGo, Brave, Mojeek) that DO NOT honour those operators; including them returns ZERO results. To bias toward an official source, just add its name as a keyword (e.g. 'python asyncio docs' or 'numpy wikipedia'), not 'site:python.org'.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "deep_research", "description": "Performs deep analysis by searching multiple sources and synthesizing a report. Use this ONLY for complex topics or if web_search fails. Do NOT use for simple factual questions (e.g. 'when was IBM founded'). CRITICAL: Keep your queries concise and keyword-focused (e.g., 'PostgreSQL 16 release notes'). DO NOT use long conversational sentences. PLAIN KEYWORDS ONLY — do NOT use search operators like 'site:', quoted \"exact phrases\", or boolean OR/AND. The search runs over Tor against scraper backends (DuckDuckGo, Brave, Mojeek) that DO NOT honour those operators; including them returns ZERO results. To bias toward an official source, add its name as a keyword (e.g. 'numpy wikipedia'), not 'site:numpy.org'.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "fact_check", "description": "Verify a complex claim using deep research and external sources. Do NOT use for simple factual questions (use web_search instead).", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "darkweb_search", "description": "Search the DARK WEB (Tor hidden services, .onion sites) via dedicated onion search engines (Ahmia, Torch, Haystak). Use this ONLY when the user explicitly wants .onion / hidden-service results — for normal questions use web_search, which is faster and broader. Returns a RANKED LIST of .onion services (title, snippet, onion URL); it does NOT open them — follow up with `browser` or `darkweb_research` to read a result. WARNING: results are UNVERIFIED hidden services and may be malicious; never trust a claim without corroboration. CRITICAL: PLAIN KEYWORDS ONLY — no 'site:', quoted phrases, or boolean OR/AND (the onion engines do not honour them). Onion engines are flaky and per-exit-node reachable, so ZERO results is normal and does NOT mean retry the same query — drop to 2-4 keywords or fall back to web_search.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "darkweb_research", "description": "Deep dark-web research: searches .onion hidden services AND fetches + synthesises the top results into a report (the dark-web analogue of deep_research). Use ONLY for complex hidden-service topics where you need the page contents, not just a list of links (for a link list use `darkweb_search`; for the clearnet use `deep_research`). Slow — each onion page is fetched over Tor. PLAIN KEYWORDS ONLY (no operators/quotes/boolean). Treat all extracted content as UNVERIFIED and corroborate before relying on it.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "update_profile", "description": "Save a permanent fact about the user (name, preferences, location). To DELETE a stored fact, pass its key with an empty value (\"\").", "parameters": {"type": "object", "properties": {"category": {"type": "string", "description": "The category for this fact (e.g., 'root', 'preferences', 'projects', 'assets', 'relationships', 'interests')."}, "key": {"type": "string"}, "value": {"type": "string", "description": "The fact to store. An empty string deletes the key from the profile."}}, "required": ["category", "key", "value"]}}},
    MANAGE_PROJECTS_TOOL_DEF,
    {"type": "function", "function": {"name": "manage_tasks", "description": "Consolidated task manager. Actions: 'create' (a TIME-based scheduled task — fires on a cron/interval clock), 'watch' (a REACTIVE task — polls a shell condition and fires only WHEN it becomes true), 'list', 'stop', 'stop_all'. Tasks persist across agent restarts; re-creating with the SAME name replaces the previous one. Use 'watch' when the trigger is a CONDITION not a time — e.g. 'when the NetMon dashboard reports the internet is down, notify me', 'when errors appear in the deploy log, investigate', 'when disk on ghost exceeds 90%, clean up'. The check_command runs in your sandbox (reaches the LAN/tailnet directly), and fires the reaction the moment it first exits 0 (edge-triggered — it won't spam while the condition stays true).", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["create", "watch", "list", "stop", "stop_all"]}, "task_name": {"type": "string", "description": "A short name for the task (required for 'create'/'watch')."}, "cron_expression": {"type": "string", "description": "For 'create': standard cron format OR 'interval:seconds' (e.g., 'interval:60'). NOTE: cron times are UTC — for a local-time schedule convert first (09:00 Athens summer = '0 6 * * *')."}, "prompt": {"type": "string", "description": "The instruction the background agent runs — for 'create' when the clock fires, for 'watch' when the condition becomes true (required for both)."}, "check_command": {"type": "string", "description": "For 'watch': a shell command that EXITS 0 when the thing to react to is TRUE (shell 'if' semantics), e.g. \"grep -q ' ERROR ' /path/app.log\", \"! curl -sf https://host/health\", \"[ $(some-metric) -gt 90 ]\". Runs in the sandbox."}, "interval_secs": {"type": "integer", "description": "For 'watch': how often (seconds, >=10) to poll the check_command."}, "task_identifier": {"type": "string", "description": "The ID of the task to kill (required for 'stop')."}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "dream_mode", "description": "Triggers Active Memory Consolidation. Use this when the user asks to 'sleep', 'rest', or 'consolidate memories'.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "self_play", "description": "Triggers the synthetic self-play training curriculum. Use this EVERY TIME the user asks to practice, train, or do self-play. It generates a completely new, random challenge in an isolated matrix. NEVER try to manually roleplay, simulate, or write code for a challenge in the main chat.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "self_play_loop", "description": "Run synthetic self-play CONTINUOUSLY, one cycle after another, in the background. Use this when the user asks to keep practicing, train until stopped, or run self-play in a loop. The loop pauses automatically when the user sends the next message, or when 'stop_self_play' is called. Do NOT use this for a single self-play cycle — use 'self_play' instead.", "parameters": {"type": "object", "properties": {"max_cycles": {"type": "integer", "description": "Optional hard cap on how many cycles to run before auto-stopping. Omit or 0 = unbounded (runs until user speaks or stop_self_play is called)."}, "model": {"type": "string", "description": "Optional upstream model override for the loop (e.g. a cheaper model). Defaults to the agent's configured model."}}, "required": []}}},
    {"type": "function", "function": {"name": "stop_self_play", "description": "Stop the currently-running continuous self-play loop started by 'self_play_loop'. No-op if no loop is running.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "list_lessons", "description": "Show LESSONS the agent has learned (mistakes-and-fixes the agent has internalized). Use this EVERY TIME the user asks 'what have you learned today / this week / so far' or asks to see the LESSON playbook / your lessons / what mistakes you've fixed. Do NOT call this for 'show me your skills' — that means the agent's TOOLS / custom skills and routes to `manage_skills`. Do NOT answer from memory — always call this tool so the list is authoritative.", "parameters": {"type": "object", "properties": {"scope": {"type": "string", "enum": ["today", "week", "all", "self_play_only"], "description": "Time / source filter. 'today' = since local midnight. 'week' = last 7 days. 'all' = every lesson on file. 'self_play_only' = only lessons whose source is self-play, no time filter. Default: 'today'."}, "limit": {"type": "integer", "description": "Max lessons to return (1-100). Default 20."}}, "required": []}}},
    {"type": "function", "function": {"name": "replan", "description": "Call this tool if your current strategy is failing or if you need to pause and rethink. It forces a fresh planning step.", "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "Why are you replanning?"}}, "required": ["reason"]}}},
    {"type": "function", "function": {"name": "abort_attempt", "description": "ESCAPE HATCH. Call this ONLY when you have proven the current task cannot be completed as specified — e.g. you have demonstrated that the validator / test has a structural bug (like `''.split('\\n')` comparing to `len(exp)==0`), the spec is internally contradictory, or a mandatory file the task depends on was never provided and cannot be generated. Do NOT call this for normal retries, for tasks you simply don't know how to solve, or to escape a hard problem — use `replan` for those. After this tool runs, the turn loop exits and the outer simulation stops retrying.", "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "Concrete, specific proof that the task is unwinnable. Name the exact line of the broken validator, the contradictory spec clauses, or the missing dependency. Vague reasons ('too hard', 'keeps failing') are not valid."}}, "required": ["reason"]}}},
    {
        "type": "function",
        "function": {
            "name": "scratchpad",
            "description": "Key/value store for short-term notes — survives across turns within the conversation. THIS is the FIRST CHOICE when the user says 'set a key', 'save a variable', 'remember X as Y for this conversation', or 'use the scratchpad'. Do NOT use file_system.write to persist tagged values — that creates orphan sandbox files that aren't recallable as named entries. action='set' stores; 'get' retrieves; 'list' shows all; 'clear' wipes everything.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["set", "get", "list", "clear"]},
                    "key": {"type": "string", "description": "The name of the variable/note (required for set/get)."},
                    "value": {"type": "string", "description": "The text/data to associate with the key (required for action='set'). Can be any string — a value, a JSON blob, an ID, etc."}
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
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Set true to authorise a destructive DROP/TRUNCATE statement (the pre-execution validator blocks these unless confirmed). Default false."
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
            "description": "List or delete the agent's SKILLS. `action='list'` returns the COMPLETE, COMPACT custom-skill inventory in ONE call — acquired tools (created via `create_skill`) AND composed macros — and is the AUTHORITATIVE answer to 'show me your skills', 'list your skills', 'what skills do you have', 'what custom tools do you have'. Call it EVERY TIME that's asked and answer FROM ITS OUTPUT: do NOT additionally reproduce every built-in tool with its full schema — that blows the token budget and truncates before the custom skills. The built-in tools are standing capabilities; summarise them by category if asked. Also handles 'forget skill X'. Do NOT call this for 'lessons learned' / 'what have you learned' / 'show me the lesson playbook' — that means lessons (mistake-and-fix entries) and routes to `list_lessons`.",
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

TOOL_DEFINITIONS.append({
    "type": "function",
    "function": {
        "name": "report_pdf",
        "description": (
            "Generate a styled multi-page PDF report and save it to the sandbox. "
            "Use this whenever the user asks for a 'report', 'whitepaper', "
            "'market analysis', 'PDF', or any document deliverable. The PDF is "
            "rendered locally (no network) and exposed via /api/download/<file>. "
            "For a DETAILED report compiled from files you already wrote (e.g. "
            "per-task .md findings), pass source_files=[...]: the tool reads and "
            "sections each file for you, so the FULL content reaches the PDF. Do "
            "NOT paste large file contents into 'sections' — that forces you to "
            "re-transcribe everything and reliably collapses into a thin summary. "
            "After this tool succeeds, include the returned markdown download "
            "link verbatim in your reply so the user can open the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Report title shown as the H1 on page 1. REQUIRED.",
                },
                "subtitle": {
                    "type": "string",
                    "description": "Optional subtitle / tagline rendered under the title.",
                },
                "author": {
                    "type": "string",
                    "description": "Optional author/byline (e.g. 'Ghost Agent').",
                },
                "sections": {
                    "type": "array",
                    "description": (
                        "Ordered list of report sections. Each section is "
                        "{heading, body} where body is markdown (paragraphs, "
                        "lists, tables, code, bold/italic). If you only have "
                        "one section you may pass a single markdown string and "
                        "the tool will wrap it for you."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string", "description": "Section heading (H2). May be empty for an unheaded prelude."},
                            "body":    {"type": "string", "description": "Section body, written in markdown."},
                        },
                        "required": ["body"],
                    },
                },
                "source_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "OPTIONAL list of sandbox file paths (markdown/text) to "
                        "compile INTO the report. The tool reads each file and "
                        "splits it into sections on its headers, so the full "
                        "detailed content reaches the PDF WITHOUT you re-typing "
                        "it. USE THIS for 'a detailed report with all findings "
                        "from all tasks' — list the task output files here "
                        "instead of pasting their content into 'sections'. You "
                        "may still pass 'sections' for an intro/summary that "
                        "leads the compiled file content."
                    ),
                },
                "filename": {
                    "type": "string",
                    "description": (
                        "OPTIONAL output filename for the PDF, e.g. "
                        "'q4_report.pdf'. Must start with an alphanumeric "
                        "and contain only letters, digits, '.', '_' or '-'. "
                        "No path separators. If omitted, a unique name like "
                        "'report_<8hex>.pdf' is generated."
                    ),
                },
            },
            "required": ["title"],
        },
    },
})

TOOL_DEFINITIONS.append({
    "type": "function",
    "function": {
        "name": "manage_composed_skills",
        "description": (
            "Group several tool calls into ONE reusable named macro (a 'composed "
            "skill') — e.g. a 'morning_briefing' that bundles weather + news "
            "headlines + system diagnostics + today's lessons into a single call. "
            "Use action='define' to create one from a list of steps, 'list' to see "
            "existing macros (including auto-discovered PROPOSED ones), 'approve' to "
            "activate a proposed macro, 'delete' to remove one. Once active, the macro "
            "is a TOP-LEVEL TOOL you invoke by its name like any built-in — its steps "
            "run and the combined results come back for you to synthesise. Default "
            "mode='parallel' fans the steps out concurrently (ideal for independent "
            "read-only steps); use mode='sequential' when a later step depends "
            "on an earlier one — a sequential step can bind its result with "
            "'save_as' and a later step can consume it as \"$name\", which is how "
            "you express a real pipeline (fetch → transform → act on the value). "
            "NOTE: a macro's steps may only call built-in tools "
            "or acquired skills — NOT other composed skills (no nesting). The dream "
            "cycle auto-proposes macros from recurring tool sequences in your history; "
            "they stay inert until you 'approve' them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["define", "list", "approve", "delete"],
                    "description": "What to do.",
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Macro name (required for define/delete). Becomes the tool "
                        "name, so use a bare identifier like 'morning_briefing' — "
                        "letters, digits and underscores only."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "For define: a short natural-language description of when "
                        "to use this macro (shown in the tool list and used for "
                        "matching)."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["parallel", "sequential"],
                    "description": (
                        "For define: 'parallel' (default) runs every step "
                        "concurrently; 'sequential' runs them in order."
                    ),
                },
                "steps": {
                    "type": "array",
                    "description": (
                        "For define: the list of tool calls this macro bundles. "
                        "Each item is an object describing one step."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {
                                "type": "string",
                                "description": (
                                    "Exact name of the tool to call in this step "
                                    "(e.g. 'system_utility', 'web_search', "
                                    "'list_lessons')."
                                ),
                            },
                            "description": {
                                "type": "string",
                                "description": "Short human label for this step (e.g. 'Local weather').",
                            },
                            "params": {
                                "type": "object",
                                "description": (
                                    "Arguments to pass to that tool (e.g. "
                                    "{\"action\": \"check_weather\"}). Use "
                                    "\"$varname\" to pull from the macro's own "
                                    "runtime parameters at call time, OR from a "
                                    "value an EARLIER step bound with 'save_as'. "
                                    "$var also interpolates inside text, e.g. "
                                    "\"Summarise this: $page_text\"."
                                ),
                            },
                            "save_as": {
                                "type": "string",
                                "description": (
                                    "Bind THIS step's result to a name that later "
                                    "steps can use as \"$name\" — this is how you "
                                    "build a real pipeline (fetch → transform → act "
                                    "on the fetched value) instead of a list of "
                                    "independent calls. Requires mode='sequential' "
                                    "(parallel steps can't see each other's "
                                    "results). Bindings only flow FORWARD."
                                ),
                            },
                            "optional": {
                                "type": "boolean",
                                "description": (
                                    "If true, this step failing does NOT fail the "
                                    "whole macro. Default false."
                                ),
                            },
                            "branch_condition": {
                                "type": "string",
                                "description": (
                                    "Sequential mode only: if this step's result "
                                    "CONTAINS this substring, jump to the "
                                    "'branch_target' sequence instead of the "
                                    "remaining steps. Pair with branch_target and "
                                    "a matching key in the top-level 'branches'."
                                ),
                            },
                            "branch_target": {
                                "type": "string",
                                "description": (
                                    "Name of the branch (a key in 'branches') to "
                                    "jump to when branch_condition matches."
                                ),
                            },
                        },
                        "required": ["tool"],
                    },
                },
                "branches": {
                    "type": "object",
                    "description": (
                        "For define (sequential mode only): alternative step "
                        "sequences, keyed by branch name. A step whose result "
                        "contains its 'branch_condition' jumps to its "
                        "'branch_target' sequence (replacing the remaining main "
                        "steps). Each value is a step list with the same shape "
                        "as 'steps'."
                    ),
                },
            },
            "required": ["action"],
        },
    },
})

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

    # Don't advertise delegate_to_swarm when no swarm cluster is configured
    # (no --swarm-nodes). Otherwise the model is shown the tool AND steered
    # into it, every call returns "Error: The Swarm Cluster is not configured"
    # (swarm.py), and that Error-prefixed result burns a strike — observed
    # live. Mirrors the image_generation gating below (schema advertised only
    # when image_gen_clients exist). The dispatch entry stays registered so a
    # hallucinated call still gets the helpful "process synchronously" steer.
    if not (context and getattr(context.llm_client, 'swarm_clients', None)):
        active_tools = [
            t for t in active_tools
            if t.get("function", {}).get("name") != "delegate_to_swarm"
        ]
    
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
                "description": "Generate a photorealistic image on the external GPU node (SD1.5 realism model). Follow 3 modes: 1) EXACT: use prompt exactly as-is, 2) ENHANCED: append photographic style/quality enhancements, 3) IMAGINATION: create a high-entropy prompt. Preserve the user's exact subject description in modes 1 and 2. LONG prompts are fully used (no truncation), and A1111 attention weights work — (sharp focus:1.2) emphasises, [background] de-emphasises. CRITICAL: If the user says the generated image is WRONG or needs fixing, DO NOT blind-guess what to change. You MUST use the `vision_analysis` tool first on the previously generated image to explicitly 'see' what went wrong.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The final image generation prompt. 1) EXACT: Provide the user's requested subject EXACTLY as they described it, without filtering or altering it. 2) ENHANCED: Append comma-separated photographic style tags; attention weights like (cinematic lighting:1.2) are supported. 3) IMAGINATION: Generate a high-entropy prompt. Detail is rewarded — the full prompt is used however long it is."
                        },
                        "steps": {
                            "type": "integer",
                            "minimum": 15,
                            "maximum": 50,
                            "description": "Inference steps. OMIT to get the node's tuned default (30). Only set it to trade quality for speed (15 = fast draft, 40+ = maximum detail)."
                        },
                        "width": {
                            "type": "integer",
                            "description": (
                                "Requested width in pixels (optional). Snapped to "
                                "the node's supported sizes: 512x768 (portrait), "
                                "544x720, 624x624 (square), 720x544, 768x512 "
                                "(landscape) — choose by aspect ratio."
                            ),
                        },
                        "height": {
                            "type": "integer",
                            "description": (
                                "Requested height in pixels (optional). Snapped "
                                "together with `width` to the node's size set."
                            ),
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Optional. Reuse the SAME seed with a tweaked prompt to refine an image the user liked; omit for a fresh random image."
                        },
                    },
                    "required": ["prompt"]
                }
            }
        })
        
    if context and getattr(context, 'sandbox_dir', None) and getattr(context, 'memory_system', None):
        try:
            import json
            # Canonical storage lives under memory_dir so skills
            # persist across sandbox wipes. Fall back to sandbox_dir
            # only when memory_dir isn't wired (early-init contexts
            # in tests etc.); the legacy-migration path inside the
            # manager handles moving any pre-existing skills over.
            _skills_base = getattr(context, 'memory_dir', None) or context.sandbox_dir
            manager = AcquiredSkillManager(
                _skills_base, context.memory_system,
                legacy_sandbox_dir=context.sandbox_dir,
            )

            # Semantic RAG Retrieval if query provided. A query FAILURE must
            # degrade to "advertise all active skills" (target_skill_names
            # stays None), NOT abort the whole advertising block — otherwise
            # a transient vector-store hiccup silently hides every acquired
            # skill from the schema the model sees while they stay
            # dispatchable (invisible-but-callable drift).
            target_skill_names = None
            if query:
                try:
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
                except Exception as rag_err:
                    logger.warning(
                        "Acquired-skill semantic routing failed (%s: %s); "
                        "advertising all active skills instead.",
                        type(rag_err).__name__, rag_err,
                    )
                    target_skill_names = None
            
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

                    # Description hardening (2026-04-24 EA incident):
                    # the LLM saw `[ACQUIRED SKILL] {desc}` but didn't
                    # realise the skill was a TOP-LEVEL TOOL. It burned
                    # 8 turns wrapping the call in `python -c`, reading
                    # `acquired_skills/<name>.py` from the sandbox (the
                    # file now lives in memory_dir, not sandbox), and
                    # writing a stub that tried to `import greece_top_news`.
                    # Fix: make the tool description aggressively
                    # explicit about invocation mode, include a concrete
                    # example using the skill's own name, and forbid the
                    # wrong patterns.
                    user_desc = skill_info.get('description', 'Acquired dynamic skill.')
                    description = (
                        f"[ACQUIRED SKILL — CALL BY NAME] {user_desc}\n\n"
                        f"USAGE: This IS a top-level tool. Invoke it directly: "
                        f"`{skill_name}(...)`. Do NOT wrap it in `execute`, "
                        f"`python -c`, or `file_system` — the implementation "
                        f"lives OUTSIDE the sandbox (in "
                        f"$GHOST_HOME/system/memory/acquired_skills/) so "
                        f"`import {skill_name}` and "
                        f"`python3 acquired_skills/{skill_name}.py` will both "
                        f"fail with ModuleNotFoundError / ENOENT. Just call "
                        f"`{skill_name}` the way you'd call any built-in tool."
                    )

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

    def _proj_ws(stateful: bool = False):
        """Working directory for file ops + code execution — see
        ``file_system.project_scoped_sandbox`` (the shared source of truth).

        When a project is active, files live under
        ``sandbox/projects/<id>/`` instead of the sandbox root, so a whole
        project's scratch space cleans up with a single
        ``rm -rf sandbox/projects/<id>``. Returns
        ``(host_dir, container_workdir)``; stateful kernel sessions opt out
        (the kernel conn file is pinned to ``/workspace``).
        """
        from .file_system import project_scoped_sandbox
        return project_scoped_sandbox(context, stateful=stateful)

    async def _run_execute(**kwargs):
        # Stateful kernel sessions can't be project-scoped (kernel conn file
        # is pinned to /workspace), so they opt out; everything else runs
        # from /workspace/projects/<id> with its host dir scoped to match.
        host_dir, workdir = _proj_ws(stateful=bool(kwargs.get("stateful")))
        return await tool_execute(
            sandbox_dir=host_dir,
            container_workdir=workdir,
            sandbox_manager=context.sandbox_manager,
            memory_dir=context.memory_dir,
            _metacog_bundle=getattr(context, "metacog", None),
            workspace_model=getattr(context, "workspace_model", None),
            **kwargs,
        )

    async def _run_browser(**kwargs):
        # Unpack the project-scoped pair from ONE _proj_ws() call. Calling it
        # twice (host_dir=_proj_ws()[0], workdir=_proj_ws()[1]) could read a
        # different current_project_id per call if a concurrent conversation
        # cleared it mid-request — desyncing the host dir from the container
        # workdir, exactly what the single-call unpack prevents.
        host_dir, workdir = _proj_ws()
        # Loopback ports of supervised sandbox services (manage_services) are
        # admitted through the browser SSRF guard so the agent can drive an
        # app it is hosting. Registry-driven; empty when none are running.
        from ..sandbox.services import active_service_ports
        return await tool_browser(
            sandbox_dir=host_dir,
            container_workdir=workdir,
            sandbox_manager=context.sandbox_manager,
            tor_proxy=context.tor_proxy,
            workspace_model=getattr(context, "workspace_model", None),
            allowed_local_ports=active_service_ports(context.sandbox_manager),
            **kwargs,
        )

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
        "file_system": lambda **kwargs: tool_file_system(sandbox_dir=_proj_ws()[0], tor_proxy=context.tor_proxy, max_context=context.args.max_context, sandbox_manager=context.sandbox_manager, read_budget=getattr(context, "_read_budget", None), **kwargs),
        "manage_services": lambda **kwargs: tool_manage_services(sandbox_manager=context.sandbox_manager, **kwargs),
        "delegate": lambda **kwargs: tool_delegate(context=context, **kwargs),
        "jobs": lambda **kwargs: tool_jobs(context=context, **kwargs),
        "notify_operator": lambda **kwargs: tool_notify_operator(context=context, **kwargs),
        "knowledge_base": lambda **kwargs: tool_knowledge_base(sandbox_dir=_proj_ws()[0], memory_system=context.memory_system, profile_memory=context.profile_memory, graph_memory=getattr(context, "graph_memory", None), llm_client=context.llm_client, model_name=getattr(context.args, "model", "default"), memory_bus=getattr(context, "memory_bus", None), episodic_memory=getattr(context, "episodic_memory", None), session_store=getattr(context, "session_store", None), **kwargs),
        "recall": lambda **kwargs: tool_recall(memory_system=context.memory_system, graph_memory=getattr(context, "graph_memory", None), **kwargs),
        "execute": _run_execute,
        "browser": _run_browser,
        "learn_skill": lambda **kwargs: tool_learn_skill(skill_memory=context.skill_memory, memory_system=context.memory_system, memory_bus=getattr(context, "memory_bus", None), **kwargs),
        "self_state": lambda **kwargs: tool_self_state(self_model=getattr(context, "self_model", None), **kwargs),
        "introspect": lambda **kwargs: tool_introspect(self_model=getattr(context, "self_model", None), context=context, **kwargs),
        "postmortem": lambda **kwargs: tool_postmortem(defect_queue=getattr(context, "defect_queue", None), **kwargs),
        "workspace": lambda **kwargs: tool_workspace(workspace_model=getattr(context, "workspace_model", None), **kwargs),
        "workspace_track": lambda **kwargs: tool_workspace_track(workspace_model=getattr(context, "workspace_model", None), **kwargs),
        "flag_uncertainty": lambda **kwargs: tool_flag_uncertainty(uncertainty_tracker=getattr(context, "uncertainty_tracker", None), **kwargs),
        "web_search": lambda **kwargs: tool_search(anonymous=context.args.anonymous, tor_proxy=context.tor_proxy, **kwargs),
        "deep_research": lambda **kwargs: tool_deep_research(anonymous=context.args.anonymous, tor_proxy=context.tor_proxy, llm_client=context.llm_client, model_name=getattr(context.args, 'model', 'default'), max_context=context.args.max_context, workspace_model=getattr(context, "workspace_model", None), **kwargs),
        "fact_check": lambda **kwargs: tool_fact_check(llm_client=context.llm_client, model_name=getattr(context.args, 'model', "qwen-3.6-35b-a3"), max_context=context.args.max_context, deep_research_callable=lambda q: tool_deep_research(query=q, anonymous=context.args.anonymous, tor_proxy=context.tor_proxy, llm_client=context.llm_client, model_name=getattr(context.args, 'model', 'default'), max_context=context.args.max_context, workspace_model=getattr(context, "workspace_model", None)), **kwargs),
        "darkweb_search": lambda **kwargs: tool_darkweb_search(anonymous=context.args.anonymous, tor_proxy=context.tor_proxy, **kwargs),
        "darkweb_research": lambda **kwargs: tool_darkweb_research(anonymous=context.args.anonymous, tor_proxy=context.tor_proxy, llm_client=context.llm_client, model_name=getattr(context.args, 'model', 'default'), max_context=context.args.max_context, workspace_model=getattr(context, "workspace_model", None), **kwargs),
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
        "postgres_admin": lambda **kwargs: tool_postgres_admin(default_uri=getattr(context.args, 'default_db', 'postgresql://ghost@127.0.0.1:5432/agent'), _metacog_bundle=getattr(context, "metacog", None), **kwargs),
        "delegate_to_swarm": lambda **kwargs: tool_delegate_to_swarm(llm_client=context.llm_client, model_name=getattr(context.args, 'model', 'default'), scratchpad=context.scratchpad, context=context, **kwargs),
        "create_skill": lambda **kwargs: tool_create_skill(sandbox_dir=context.sandbox_dir, memory_dir=getattr(context, "memory_dir", None), memory_system=context.memory_system, sandbox_manager=context.sandbox_manager, **kwargs),
        "manage_skills": lambda **kwargs: tool_manage_skills(sandbox_dir=context.sandbox_dir, memory_dir=getattr(context, "memory_dir", None), memory_system=context.memory_system, **kwargs),
        "manage_composed_skills": lambda **kwargs: tool_manage_composed_skills(
            context=context,
            # Derive the known-tool set at CALL time from the fully-populated
            # dispatch table, so it includes the dynamically-appended tools
            # (vision_analysis, image_generation) AND acquired skills — a
            # composed-skill step that calls an acquired skill otherwise got
            # a spurious "not a recognised built-in" warning.
            known_tools=(set(tools.keys()) | {"vision_analysis", "image_generation"}),
            **kwargs,
        ),
    }
    
    from .vision import tool_vision_analysis
    tools["vision_analysis"] = lambda **kwargs: tool_vision_analysis(llm_client=context.llm_client, sandbox_dir=_proj_ws()[0], tor_proxy=context.tor_proxy, **kwargs)

    from .report_pdf import tool_generate_pdf
    tools["report_pdf"] = lambda **kwargs: tool_generate_pdf(sandbox_dir=_proj_ws()[0], **kwargs)

    if getattr(context.llm_client, 'image_gen_clients', None):
        from .image_gen import tool_generate_image
        tools["image_generation"] = lambda **kwargs: tool_generate_image(llm_client=context.llm_client, sandbox_dir=_proj_ws()[0], **kwargs)
        
    if context and getattr(context, 'sandbox_dir', None) and getattr(context, 'memory_system', None):
        try:
            _skills_base = getattr(context, 'memory_dir', None) or context.sandbox_dir
            manager = AcquiredSkillManager(
                _skills_base, context.memory_system,
                legacy_sandbox_dir=context.sandbox_dir,
            )
            skills = manager.get_all_skills()
            
            _BUILTIN_TOOL_NAMES = frozenset(tools.keys())
            for skill_name, skill_info in skills.items():
                if skill_info.get("status") == "active":
                    if skill_name in _BUILTIN_TOOL_NAMES:
                        logger.warning(
                            f"Acquired skill '{skill_name}' shadows a built-in tool — skipping."
                        )
                        continue
                    def make_skill_runner(name=skill_name, _mgr=manager):
                        async def _run(**kwargs):
                            import json
                            args_str = json.dumps(kwargs)

                            logger.info(f"Executing Acquired Skill: {name}")
                            pretty_log("Executing Skill", f"Running custom tool: {name}", icon=Icons.TOOL_CODE)

                            # Canonical skill file lives under memory_dir
                            # (outside the sandbox). Read it, then pass
                            # `content=` to tool_execute so the execution
                            # still happens inside the sandbox — the
                            # source of truth stays safe across sandbox
                            # wipes. If the file is missing (deleted out
                            # from under us, manager/registry drift), we
                            # report a clear error instead of running a
                            # stale sandbox copy.
                            try:
                                canonical_path = _mgr.skills_dir / f"{name}.py"
                                skill_src = canonical_path.read_text(encoding="utf-8")
                            except FileNotFoundError:
                                msg = (
                                    f"Acquired skill '{name}' source file not found at "
                                    f"{canonical_path}. The registry entry is stale; call "
                                    f"manage_skills(action='delete', skill_name='{name}') "
                                    f"or re-create it via create_skill."
                                )
                                logger.error(msg)
                                pretty_log("Skill Missing", msg, level="ERROR", icon=Icons.FAIL)
                                return msg
                            except Exception as e:
                                msg = f"Could not read acquired skill {name}: {type(e).__name__}: {e}"
                                logger.error(msg)
                                return msg

                            try:
                                result = await tool_execute(
                                    sandbox_dir=context.sandbox_dir,
                                    sandbox_manager=context.sandbox_manager,
                                    memory_dir=getattr(context, "memory_dir", None),
                                    filename=f"acquired_skills/{name}.py",
                                    content=skill_src,
                                    args=[args_str]
                                )
                                # Telemetry: tool_execute returns error
                                # strings rather than raising, so classify
                                # the RESULT. Logging success=True
                                # unconditionally let broken skills reset
                                # their failure_count and dodge retirement.
                                ok = _acquired_skill_result_ok(result)
                                manager.log_telemetry(name, success=ok)
                                if ok:
                                    logger.info(f"Acquired Skill '{name}' executed successfully.")
                                else:
                                    logger.warning(f"Acquired Skill '{name}' returned a failure result.")
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

    # Wire a runner for every registered composed skill (macro) so the names
    # advertised by register_composed_skills in get_active_tool_definitions
    # are actually dispatchable here. Mirrors the acquired-skill loop above;
    # non-fatal — a failure just means macros aren't callable this build.
    try:
        register_composed_skill_runners(tools, context)
    except Exception as e:
        logger.debug(f"Composed skill runner wiring failed: {type(e).__name__}: {e}")

    return tools
