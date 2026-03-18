# src/ghost_agent/core/agent.py

import asyncio
import datetime
import json
import logging
import uuid
import re
import sys
import gc

import ctypes
import platform
import httpx
from typing import List, Dict, Any, Optional
from pathlib import Path

from .prompts import SYSTEM_PROMPT, CODE_SYSTEM_PROMPT, SMART_MEMORY_PROMPT, PLANNING_SYSTEM_PROMPT, DBA_SYSTEM_PROMPT, SYSTEM_3_GENERATION_PROMPT, SYSTEM_3_EVALUATOR_PROMPT
from .planning import TaskTree, TaskStatus
from ..utils.logging import Icons, pretty_log, request_id_context
from ..utils.token_counter import estimate_tokens
from ..tools.registry import get_available_tools, TOOL_DEFINITIONS, get_active_tool_definitions
from ..tools.tasks import tool_list_tasks
from ..memory.skills import SkillMemory

logger = logging.getLogger("GhostAgent")

def extract_json_from_text(text: str) -> dict:
    """Safely extracts JSON from LLM outputs, ignoring conversational filler and markdown blocks."""
    import re, json, ast
    # Qwen Syntax Healing: Fix {"name"="tool"...} or {"name"= "tool"...} hallucinations
    text = re.sub(r'{\s*"name"\s*=\s*(["\'])', r'{"name": \1', text)
    
    def _parse(t):
        try:
            return json.loads(t, strict=False)
        except json.JSONDecodeError:
            try:
                # AST Fallback for models that output Python dicts instead of strict JSON
                pt = t.replace("true", "True").replace("false", "False").replace("null", "None")
                res = ast.literal_eval(pt)
                if isinstance(res, dict): return res
            except Exception as e:
                logger.debug(f"JSON AST fallback failed: {type(e).__name__}")
            return {}

    try:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
        if match: 
            p = _parse(match.group(1))
            if p: return p
            
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1: 
            p = _parse(text[start:end+1])
            if p: return p
            
        return _parse(text)
    except Exception as e:
        logger.debug(f"JSON extraction failed: {type(e).__name__}")
        return {}

class GhostContext:
    def __init__(self, args, sandbox_dir, memory_dir, tor_proxy):
        self.args = args
        self.sandbox_dir = sandbox_dir
        self.memory_dir = memory_dir
        self.tor_proxy = tor_proxy
        self.llm_client = None
        self.memory_system = None
        self.profile_memory = None
        self.skill_memory = None
        self.scratchpad = None
        self.sandbox_manager = None
        self.scheduler = None
        self.last_activity_time = datetime.datetime.now()
        self.cached_sandbox_state = None

class GhostAgent:
    def __init__(self, context: GhostContext):
        self.context = context
        self.available_tools = get_available_tools(context)
        self.agent_semaphore = asyncio.Semaphore(10)
        self.memory_semaphore = asyncio.Semaphore(1)

    def release_unused_ram(self):
        try:
            gc.collect()
            if platform.system() == "Linux":
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except: pass
        except: pass

    def clear_session(self):
        if hasattr(self.context, 'scratchpad') and self.context.scratchpad:
            self.context.scratchpad.clear()
        self.release_unused_ram()
        return True

    def _prepare_planning_context(self, tools_run_this_turn: List[Dict[str, Any]]) -> str:
        if not tools_run_this_turn:
            return "None (Start of Task)"
        
        char_limit = max(4000, int(self.context.args.max_context * 3.5 * 0.1))
        
        outputs = []
        for t in tools_run_this_turn:
            content = str(t.get("content", ""))
            if len(content) > char_limit:
                # Keep top char_limit so the Planner actually sees the search matches
                content = content[:char_limit] + "\n\n... [TRUNCATED: Tool output too long. Showing top results only.]"
            outputs.append(f"Tool [{t.get('name', 'unknown')}]: {content}")
            
        return "\n\n".join(outputs)

    def _get_recent_transcript(self, messages: List[Dict[str, Any]]) -> str:
        msg_limit = max(40, int(self.context.args.max_context / 500))
        char_limit = max(500, int(self.context.args.max_context * 3.5 * 0.02))
        
        recent_transcript = ""
        transcript_msgs = [m for m in messages if m.get("role") in ["user", "assistant", "tool"]][-msg_limit:]
        for m in transcript_msgs:
            content_val = m.get('content') or ""
            if isinstance(content_val, list):
                text_parts = []
                for item in content_val:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            text_parts.append("[Image attached and passed to vision node]")
                content_val = "\n".join(text_parts)
            content_str = str(content_val)
            
            role = m['role'].upper()
            if role == "TOOL":
                role = f"TOOL ({m.get('name', 'unknown')})"
            recent_transcript += f"{role}: {content_str[:char_limit]}\n"
        return recent_transcript

    def process_rolling_window(self, messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        if not messages: return []
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if len(system_msgs) > 1:
            merged_content = "\n\n".join([str(m.get("content", "")) for m in system_msgs])
            system_msgs = [{"role": "system", "content": merged_content}]
        raw_history = [m for m in messages if m.get("role") != "system"]
        
        current_tokens = sum(estimate_tokens(str(m.get("content", ""))) for m in system_msgs)
        final_history = []
        
        # Pure sliding window from newest to oldest. 
        # We NEVER mutate historical strings, we just drop the oldest ones if we run out of space.
        for msg in reversed(raw_history):
            msg_tokens = estimate_tokens(str(msg.get("content", "")))
            if current_tokens + msg_tokens > max_tokens: 
                break
            final_history.append(msg)
            current_tokens += msg_tokens
            
        final_history.reverse()
        return system_msgs + final_history
        
    async def _prune_context(self, messages: List[Dict[str, Any]], max_tokens: int = 12000, model: str = "test-model") -> List[Dict[str, Any]]:
        current_tokens = sum(estimate_tokens(str(m.get("content", ""))) for m in messages)
        if current_tokens < max_tokens:
            return messages
            
        pretty_log("Context Optimization", f"Context hit {current_tokens} tokens (> {max_tokens}). Running summarization pass...", icon=Icons.CUT)
        
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]

        # If we have very few messages but still hit the token limit, 
        # just truncate without summarizing as it's likely a huge prompt
        if len(non_system_msgs) <= 5:
            truncated = non_system_msgs[-3:]
            # Scrub images from the truncated fallback
            scrubbed_truncated = []
            for msg in truncated:
                if isinstance(msg.get("content"), list):
                    new_content = []
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "image_url":
                            new_content.append({"type": "text", "text": "[Image attached and passed to vision node]"})
                        else:
                            new_content.append(block)
                    scrubbed_truncated.append({**msg, "content": new_content})
                else:
                    scrubbed_truncated.append(msg)
            return system_msgs + scrubbed_truncated

        print("DEBUG PRUNE: doing summarization!")
        # Keep recent context (last 3 turns = 6 messages + goal)
        original_goal = non_system_msgs[0]
        recent_context = non_system_msgs[-6:] # Keep last ~3 turns intact
        
        middle_messages = non_system_msgs[1:-6]
        
        if not middle_messages:
            return system_msgs + [original_goal] + recent_context
            
        # Condense the middle messages using a fast LLM worker
        condense_prompt = "The following is the middle segment of a long conversational transcript between an AI Agent and a User. Summarize the key actions taken, facts learned, and the current state of progress. Be concise. DO NOT write code. ONLY output the summary.\n\nTRANSCRIPT:\n"
        for m in middle_messages:
            content_val = m.get('content') or ""
            if isinstance(content_val, list):
                text_parts = []
                for item in content_val:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            text_parts.append("[Image attached and passed to vision node]")
                content_val = "\n".join(text_parts)
            content_str = str(content_val)
            condense_prompt += f"{m.get('role').upper()}:\n{content_str[:4000]}\n---\n"
            
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": condense_prompt}],
            "temperature": 0.0,
            "max_tokens": 800
        }
        
        summary = "[SYSTEM: PREVIOUS TURNS SUMMARIZED]\n\n"
        try:
            summary_data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
            summary += str(summary_data["choices"][0]["message"].get("content") or "No summary generated.")
        except Exception as e:
            logger.warning(f"Context summarization failed: {e}")
            summary += f"(Summarization unavailable due to error, dropping old turns: {e})"
            
        return system_msgs + [original_goal, {"role": "user", "content": summary}] + recent_context

    async def process_journal_queue(self):
        if not hasattr(self.context, 'journal'): return

        items = await asyncio.to_thread(self.context.journal.pop_all)
        if not items: return
        
        pretty_log("Hippocampus", f"Waking up to process {len(items)} buffered memories...", icon=Icons.BRAIN_THINK)
        
        processed = 0
        for i, item in enumerate(items):
            idle_secs = (datetime.datetime.now() - self.context.last_activity_time).total_seconds()
            if idle_secs < 30:
                pretty_log("Hippocampus", f"User returned! Suspending memory processing. ({len(items)-i} items left)", icon=Icons.STOP)
                await asyncio.to_thread(self.context.journal.push_front, items[i:])
                break

            try:
                if item["type"] == "smart_memory":
                    await self.run_smart_memory_task(item["data"]["text"], item["data"]["model"], self.context.args.smart_memory)
                elif item["type"] == "post_mortem":
                    await self._execute_post_mortem(item["data"]["user"], item["data"]["tools"], item["data"]["ai"], item["data"]["model"])
                processed += 1
            except Exception as e:
                import logging
                logging.getLogger("GhostAgent").error(f"Journal processing error: {e}")
            await asyncio.sleep(0.5)
            
        if processed > 0:
            pretty_log("Hippocampus", f"Successfully consolidated {processed} memories.", icon=Icons.OK)

    async def run_smart_memory_task(self, interaction_context: str, model_name: str, selectivity: float):
        if not self.context.memory_system: return
        
        # --- ⚡ FAST-ABORT HEURISTIC (ZERO LLM COMPUTE) ---
        import re
        # We only care if the user said something memorable. Extract user lines.
        user_lines = [line.split("USER:", 1)[-1] for line in interaction_context.splitlines() if line.startswith("USER:")]
        user_text = " ".join(user_lines).lower() if user_lines else interaction_context.lower()
        
        # Fast exit if no identity/preference keywords are present
        if not re.search(r'\b(i|me|my|mine|prefer|always|never|remember|project|build|work|name|live|use|hate|love|want|need)\b', user_text):
            return
            

        async with self.memory_semaphore:
            interaction_context = interaction_context.encode('utf-8', 'replace').decode('utf-8').replace("\r", "")
            # Strict control character scrubbing for C++ JSON parsers
            interaction_context = "".join(ch for ch in interaction_context if ord(ch) >= 32 or ch in "\n\t")
            ic_lower = interaction_context.lower()
            summary_triggers = ["summarize", "summary", "recall", "tell me about", "what is", "recap", "forget", "list documents"]
            is_requesting_summary = any(w in ic_lower for w in summary_triggers)
            
            if is_requesting_summary and len(interaction_context) > 1500:
                return
                
            final_prompt = SMART_MEMORY_PROMPT + f"\n\n### EPISODE LOG:\n{interaction_context}"
            try:
                payload = {"model": model_name, "messages": [{"role": "user", "content": final_prompt}], "stream": False, "temperature": 0.1, "max_tokens": 1024}
                data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
                content = data["choices"][0]["message"]["content"]
                result_json = extract_json_from_text(content)
                score, fact, profile_up = float(result_json.get("score", 0.0)), result_json.get("fact", ""), result_json.get("profile_update", None)
                
                if fact is None: fact = ""
                fact_lc = fact.lower()
                is_personal = any(w in fact_lc for w in ["user", "me", "my ", " i ", "identity", "preference", "like"])
                is_technical = any(w in fact_lc for w in ["file", "path", "code", "error", "script", "project", "repo", "build", "library", "version"])
                
                if score >= selectivity and fact and len(fact) <= 200 and len(fact) >= 5 and "none" not in fact_lc:
                    if score >= 0.9 and not (is_personal or is_technical):
                        pretty_log("Auto Memory Skip", f"Discarded generic knowledge: {fact}", icon=Icons.STOP)
                        return
                    memory_type = "identity" if (score >= 0.9 and profile_up) else "auto"
                    
                    # --- CONTRADICTION ENGINE (LLM-Driven Belief Revision) ---
                    try:
                        candidates = await asyncio.to_thread(self.context.memory_system.search_advanced, fact, limit=3)
                        ids_to_delete = []
                        old_facts = []
                        
                        if candidates:
                            for c in candidates:
                                if c.get('score', 1.0) < 0.6: # Broad threshold to catch potential semantic collisions
                                    old_facts.append({"id": c['id'], "text": c['text']})
                                    
                        if old_facts:
                            eval_prompt = f"NEW FACT:\n{fact}\n\nOLD FACTS:\n" + "\n".join([f"ID: {f['id']} | TEXT: {f['text']}" for f in old_facts]) + "\n\nAnalyze if the NEW FACT contradicts, updates, or supersedes any OLD FACTS. Return ONLY a JSON object with a list of 'ids' to delete. If they safely coexist (e.g. they refer to different topics/projects), return an empty list.\n\nExample: {{\"ids\": [\"ID:123\"]}}"
                            eval_payload = {"model": model_name, "messages": [{"role": "system", "content": "You are a Belief Revision Engine. Output JSON."}, {"role": "user", "content": eval_prompt}], "temperature": 0.0, "max_tokens": 1024}
                            eval_data = await self.context.llm_client.chat_completion(eval_payload, use_worker=True, is_background=True)
                            eval_res = extract_json_from_text(eval_data["choices"][0]["message"]["content"])
                            
                            raw_ids = eval_res.get("ids", [])
                            ids_to_delete = [str(i).replace("ID: ", "").replace("ID:", "").strip() for i in raw_ids]
                            
                        if ids_to_delete:
                            await asyncio.to_thread(self.context.memory_system.collection.delete, ids=ids_to_delete)
                            pretty_log("Belief Revision", f"Erased {len(ids_to_delete)} outdated/contradicting memories.", icon=Icons.CUT)
                            
                    except Exception as ce:
                        logger.error(f"Contradiction Engine error: {ce}")
                        
                    # Save the new fact (bypassing the old simplistic smart_update math check, since we just logically validated it)
                    from ..utils.helpers import get_utc_timestamp
                    await asyncio.to_thread(self.context.memory_system.add, fact, {"timestamp": get_utc_timestamp(), "type": memory_type})
                    pretty_log("Auto Memory Store", f"[{score:.2f}] {fact}", icon=Icons.MEM_SAVE)
                    
                    if memory_type == "identity" and self.context.profile_memory:
                        await asyncio.to_thread(
                            self.context.profile_memory.update,
                            profile_up.get("category", "notes"), 
                            profile_up.get("key", "info"), 
                            profile_up.get("value", fact)
                        )
            except Exception as e: logger.error(f"Smart memory task failed: {e}")

    async def _execute_post_mortem(self, last_user_content: str, tools_run: list, final_ai_content: str, model: str):
        try:
            history_summary = f"User: {last_user_content}\n"
            for t_msg in tools_run[-5:]:
                history_summary += f"Tool {t_msg.get('name', 'unknown')}: {str(t_msg.get('content', ''))[:200]}\n"
                
            # Aggressively strip lone surrogates and raw control characters for C++ backends
            def _clean_for_cpp(text: str) -> str:
                if not isinstance(text, str): return str(text)
                text = text.encode('utf-8', 'replace').decode('utf-8')
                return "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t\r")

            last_user_content = _clean_for_cpp(last_user_content)
            final_ai_content = _clean_for_cpp(final_ai_content)
            history_summary = _clean_for_cpp(history_summary)
            
            learn_prompt = f"### TASK POST-MORTEM\nReview this interaction. The agent either struggled and succeeded, OR failed completely. Identify the core technical error, hallucination, or bad strategy. Extract a concrete rule to fix or avoid this in the future.\n\nHISTORY:\n{history_summary}\n\nFINAL AI: {final_ai_content[:500]}\n\nReturn ONLY a JSON object with 'task', 'mistake', and 'solution' (what to do instead next time/the anti-pattern to avoid). If no unique technical lesson is found, return null."
            
            payload = {"model": model, "messages": [{"role": "system", "content": "You are a Meta-Cognitive Analyst. Output JSON."}, {"role": "user", "content": learn_prompt}], "temperature": 0.1, "max_tokens": 1024}
            l_data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
            l_content = str(l_data["choices"][0]["message"].get("content") or "")
            if l_content and "null" not in l_content.lower():
                l_json = extract_json_from_text(l_content)
                if all(k in l_json for k in ["task", "mistake", "solution"]):
                    if getattr(self.context, 'skill_memory', None):
                        await asyncio.to_thread(
                            self.context.skill_memory.learn_lesson,
                            l_json["task"], l_json["mistake"], l_json["solution"],
                            memory_system=self.context.memory_system
                        )
                    pretty_log("Auto-Learning", "New lesson captured automatically", icon=Icons.IDEA)
        except Exception as e:
            logger.error(f"Post-mortem failed: {e}")

    async def handle_chat(self, body: Dict[str, Any], background_tasks, request_id: Optional[str] = None):
        req_id = request_id or str(uuid.uuid4())[:8]
        token = request_id_context.set(req_id)
        self.context.last_activity_time = datetime.datetime.now()
        
        try:
            async with self.agent_semaphore:
                char_budget = int(self.context.args.max_context * 3.5)
                pretty_log("Request Initialized", special_marker="BEGIN")
                messages, model, stream_response = body.get("messages", []), body.get("model", "qwen-3.5-9b"), body.get("stream", False)
                
                if len(messages) > 500:
                    messages = [m for m in messages if m.get("role") == "system"] + messages[-500:]
                for m in messages:
                    if isinstance(m.get("content"), str): m["content"] = m["content"].replace("\r", "")
                
                last_user_content = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
                lc = last_user_content.lower()
                
                coding_keywords = [r"\bpython\b", r"\bbash\b", r"\bsh\b", r"\bscript\b", r"\bcode\b", r"\bdef\b", r"\bimport\b", r"\bhtml\b", r"\bcss\b", r"\bjs\b", r"\bjavascript\b", r"\btypescript\b", r"\breact\b", r"\bweb\b", r"\bfrontend\b"]
                coding_actions = [r"\bwrite\b", r"\brun\b", r"\bexecute\b", r"\bdebug\b", r"\bfix\b", r"\bcreate\b", r"\bgenerate\b", r"\bcount\b", r"\bcalculate\b", r"\banalyze\b", r"\bscrape\b", r"\bplot\b", r"\bgraph\b", r"\bbuild\b", r"\bdevelop\b"]
                has_coding_intent = False
                
                if any(re.search(k, lc) for k in coding_keywords):
                    if any(re.search(a, lc) for a in coding_actions): 
                        has_coding_intent = True
                if any(ext in lc for ext in [".py", ".js", ".html", ".css", ".ts", ".tsx", ".jsx", ".sh"]) or re.search(r'\bscript\b', lc): 
                    has_coding_intent = True
                
                dba_keywords = [r"\bsql\b", r"\bpostgres\b", r"\bpostgresql\b", r"\bpsql\b", r"\bdatabase\b", r"\bpg_stat\b", r"\bexplain analyze\b", r"\bquery\b", r"\bcte\b", r"\brdbms\b", r"\bdba\b", r"\bschema\b", r"\bvacuum\b", r"\bmvcc\b"]
                has_dba_intent = any(re.search(k, lc) for k in dba_keywords)
                
                meta_keywords = [r"\btitle\b", r"\bname this\b", r"\brename\b", r"\bsummary\b", r"\bsummarize\b", r"\bcaption\b", r"\bdescribe\b"]
                is_meta_task = any(re.search(k, lc) for k in meta_keywords)
                if re.match(r'^[\d\s\+\-\*\/\(\)\=\?]+$', lc):
                    has_coding_intent = False
                    
                profile_context = await asyncio.to_thread(self.context.profile_memory.get_context_string) if self.context.profile_memory else ""
                profile_context = profile_context.replace("\r", "")
                
                working_memory_context = ""
                


                base_prompt = SYSTEM_PROMPT.replace("{{PROFILE}}", profile_context)
                
                # Dynamic "Perfect It" Protocol Injection
                if getattr(self.context.args, 'perfect_it', False):
                    # Inject as item 5 before Tool Orchestration
                    last_tool_content = "None"
                    if locals().get('tools_run_this_turn') and len(tools_run_this_turn) > 0:
                        last_tool_content = str(tools_run_this_turn[-1].get('content', ''))[:15000]
                    
                    base_prompt = base_prompt.replace(
                        "### TOOL ORCHESTRATION", 
                        f'5. THE "PERFECT IT" PROTOCOL: Upon successfully completing a complex technical task, analyze the result (Last Tool Output: {last_tool_content}) and proactively suggest one concrete way to optimize it.\n\n### TOOL ORCHESTRATION'
                    )
                base_prompt += working_memory_context
                
                # QWEN OPTIMIZATION: Generate Tool XML Schema ONCE to preserve KV Cache
                from .prompts import QWEN_TOOL_PROMPT
                all_tools = get_active_tool_definitions(self.context)
                funcs_only = [t["function"] for t in all_tools]
                minified_schemas = json.dumps(funcs_only, separators=(',', ':'))
                base_prompt += f"\n\n{QWEN_TOOL_PROMPT.replace('{tool_schemas}', minified_schemas)}\n"
                
                base_prompt = base_prompt.replace("\r", "")
                
                active_persona = ""
                if has_dba_intent and not is_meta_task:
                    current_temp = self.context.args.temperature
                    pretty_log("Mode Switch", "Ghost PostgreSQL DBA Activated", icon=Icons.MODE_GHOST)
                    active_persona = f"{DBA_SYSTEM_PROMPT.replace('{{PROFILE}}', profile_context)}\n\n"
                elif has_coding_intent:
                    current_temp = self.context.args.temperature
                    pretty_log("Mode Switch", "Ghost Python Specialist Activated", icon=Icons.MODE_GHOST)
                    active_persona = f"{CODE_SYSTEM_PROMPT.replace('{{PROFILE}}', profile_context)}\n\n"
                else:
                    current_temp = self.context.args.temperature

                # base_prompt += active_persona  <-- RELOCATED to user message for cache efficacy
                
                found_system = False
                for m in messages:
                    if m.get("role") == "system": m["content"] = base_prompt; found_system = True; break
                if not found_system: messages.insert(0, {"role": "system", "content": base_prompt})
                
                is_fact_check = "fact-check" in lc or "verify" in lc
                
                tool_action_verbs = [
                    "search", "download", "run", "execute", "schedule", "read", "fetch", 
                    "calculate", "count", "summarize", "find", "open", "check", "test",
                    "delete", "remove", "rename", "move", "copy", "scrape", "ingest"
                , "create", "draw", "make", "generate", "picture", "image", "paint"]
                has_action_verb = any(v in lc for v in tool_action_verbs)
                
                is_conversational = not has_coding_intent and not has_dba_intent and not is_meta_task and not has_action_verb
                
                # OPTIMIZATION: Detect trivial greetings to bypass heavy processing
                is_trivial_greeting = (
                    is_conversational and
                    len(lc.split()) <= 5 and
                    "remember" not in lc and
                    "previous" not in lc
                )
                
                should_fetch_memory = (
                    not is_fact_check and
                    not is_trivial_greeting and
                    (not has_coding_intent or "remember" in last_user_content or "previous" in last_user_content)
                )
                
                fetched_mem_context = ""
                if self.context.memory_system and last_user_content and should_fetch_memory:

                    mem_context = await asyncio.to_thread(self.context.memory_system.search, last_user_content)
                    if mem_context:
                        mem_context = mem_context.replace("\r", "")
                        pretty_log("Memory Context", f"Retrieved for: {last_user_content}", icon=Icons.BRAIN_CTX)
                        fetched_mem_context = f"### MEMORY CONTEXT:\n{mem_context}\n\n"
                        
                fetched_playbook = ""  # Now dynamically populated inside the loop
                                        
                messages = self.process_rolling_window(messages, self.context.args.max_context)
                
                final_ai_content, created_time = "", int(datetime.datetime.now().timestamp())
                force_stop, seen_tools, tool_usage, last_was_failure = False, set(), {}, False
                raw_tools_called = set()
                execution_failure_count = 0
                tools_run_this_turn = []
                forget_was_called = False
                thought_content = ""
                was_complex_task = False
                
                task_tree = TaskTree()
                current_plan_json = {}
                
                current_plan_json = {}
                force_final_response = False

                for turn in range(20):
                    self.context.last_activity_time = datetime.datetime.now() # Heartbeat
                    
                    turn_is_conversational = is_conversational and turn == 0

                    if turn > 2: was_complex_task = True
                    if force_stop: break
                    
                    scratch_data = self.context.scratchpad.list_all() if getattr(self.context, 'scratchpad', None) else "None."
                    if has_coding_intent:
                        if self.context.cached_sandbox_state is None:
                            from ..tools.file_system import tool_list_files
                            params = {
                                "sandbox_dir": self.context.sandbox_dir, 
                                "memory_system": self.context.memory_system
                            }
                            sandbox_state = await tool_list_files(**params)
                            self.context.cached_sandbox_state = sandbox_state
                        else:
                            sandbox_state = self.context.cached_sandbox_state
                    else:
                        sandbox_state = "N/A"
                    
                    # Use System 2 Planner based on context arguments (Mock-safe check)
                    use_plan = getattr(self.context.args, 'use_planning', False) == True
                    if use_plan and not turn_is_conversational:
                        pretty_log("Reasoning Loop", f"Turn {turn+1} Strategic Analysis...", icon=Icons.BRAIN_PLAN)
                        
                        last_tool_output = self._prepare_planning_context(tools_run_this_turn[-2:])
                        recent_transcript = self._get_recent_transcript(messages)
                            
                        tool_hints = {
                            "system_utility": "weather, health",
                            "execute": "python, bash",
                            "postgres_admin": "sql"
                        }
                        available_tools_list = ", ".join([
                            f"{t['function']['name']} ({tool_hints.get(t['function']['name'], 'native tool')})"
                            for t in get_active_tool_definitions(self.context)
                        ])
                        state_limit = max(1500, int(char_budget * 0.05))
                        safe_scratch = str(scratch_data)
                        if len(safe_scratch) > state_limit: safe_scratch = safe_scratch[:state_limit] + "\n...[TRUNCATED]"
                        safe_sandbox = str(sandbox_state)
                        if len(safe_sandbox) > state_limit: safe_sandbox = safe_sandbox[:state_limit] + "\n...[TRUNCATED]"

                        planner_transient = f"""
### CURRENT SITUATION
SCRAPBOOK:
{safe_scratch}
SANDBOX STATE:
{safe_sandbox if has_coding_intent else 'N/A'}

User Request: {last_user_content}
Last Tool Output: {last_tool_output}

### AVAILABLE NATIVE TOOLS
[{available_tools_list}]
CRITICAL INSTRUCTION: If an action requires a tool, explicitly name the native JSON tool you intend to use. DO NOT plan to write Python scripts for tasks that have a dedicated native tool. If the user is just asking a question or requesting a code/SQL explanation, set "next_action_id" to "none" and do NOT plan to use a tool.

### TEMPORAL ANCHOR (READ CAREFULLY)
You are currently at TURN {turn+1}. Trust your CURRENT PLAN JSON to know what is already DONE. NEVER revert a 'DONE' task back to 'PENDING'.

### CURRENT PLAN (JSON)
{json.dumps(current_plan_json, indent=2) if current_plan_json else "No plan yet."}
"""
                        planner_messages = [
                            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                            {"role": "user", "content": f"### RECENT CONVERSATION:\n{recent_transcript}\n\n{planner_transient.strip()}"}
                        ]
                        
                        planning_payload = {
                            "model": model,
                            "messages": planner_messages,
                            "temperature": 0.0,
                            "top_p": 0.1,
                            "max_tokens": 1024,
                            "response_format": {"type": "json_object"}
                        }
                        
                        try:
                            p_data = await self.context.llm_client.chat_completion(planning_payload, use_swarm=True)
                            plan_content = p_data["choices"][0]["message"].get("content", "")
                            plan_json = extract_json_from_text(plan_content)
                            
                            thought_content = plan_json.get("thought", "No thought provided.")
                            tree_update = plan_json.get("tree_update", {})
                            next_action_id = plan_json.get("next_action_id", "")
                            required_tool = plan_json.get("required_tool", "all")
                            
                            if tree_update:
                                task_tree.load_from_json(tree_update)
                                current_plan_json = task_tree.to_json()
                                
                            tree_render = task_tree.render()
                            
                            # Planning content is no longer injected into history messages
                            
                            pretty_log("INTERNAL MONOLOGUE", icon=Icons.BRAIN_THINK, special_marker="SECTION_START")
                            pretty_log("Planner Monologue", thought_content, icon=Icons.BRAIN_THINK)
                            pretty_log("INTERNAL MONOLOGUE", icon=Icons.BRAIN_THINK, special_marker="SECTION_END")
                            pretty_log("Reasoning Loop", f"Plan Updated. Focus: {next_action_id}", icon=Icons.OK)
                            
                            if task_tree.root_id and task_tree.nodes[task_tree.root_id].status == TaskStatus.DONE and turn > 0:
                                pretty_log("Finalizing", "Agent signaled completion", icon=Icons.OK)
                                force_stop = True
                        except Exception as e:
                            logger.error(f"Planning step failed: {e}")
                            if not any("### ACTIVE STRATEGY" in m.get("content", "") for m in messages):
                                messages.append({"role": "user", "content": "### ACTIVE STRATEGY: Proceed directly to using a tool. Do NOT provide any conversational response this turn, only output a tool_calls array!"})

                    # Dynamic state no longer mutated via re.sub

                    if last_was_failure:
                        if execution_failure_count == 1:
                            active_temp = max(current_temp, 0.40)
                        elif execution_failure_count >= 2:
                            active_temp = max(current_temp, 0.60)
                        else:
                            active_temp = min(current_temp + 0.1, 0.80)
                        pretty_log("Brainstorming", f"Adjusting variance to {active_temp:.2f} to solve error", icon=Icons.IDEA)
                    else:
                        active_temp = current_temp
                        
                    if turn_is_conversational and active_temp < 0.7:
                        active_temp = 0.7

                    # Proactive Context Pruning before request
                    messages = await self._prune_context(messages, max_tokens=self.context.args.max_context, model=model)
                    
                    # --- INTENT-DRIVEN SKILL RECALL ---
                    fetched_playbook = ""
                    if self.context.skill_memory:
                        skill_query = last_user_content
                        if use_plan and not turn_is_conversational and locals().get("required_tool", "none") not in ["none", "all"]:
                            skill_query = f"Tool: {required_tool} - Context: {thought_content}"
                        playbook = await asyncio.to_thread(self.context.skill_memory.get_playbook_context, query=skill_query, memory_system=self.context.memory_system)
                        if playbook:
                            fetched_playbook = f"### SKILL PLAYBOOK:\n{playbook}\n\n"

                    dynamic_state = f"### DYNAMIC SYSTEM STATE\nCURRENT TIME: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Day: {datetime.datetime.now().strftime('%A')})\n\nSCRAPBOOK:\n{scratch_data}\n\n"
                    if has_coding_intent:
                        dynamic_state += f"CURRENT SANDBOX STATE:\n{sandbox_state}\n\n"
                    if use_plan and not turn_is_conversational and 'thought_content' in locals() and thought_content:
                        dynamic_state += f"ACTIVE STRATEGY & PLAN:\nTHOUGHT: {thought_content}\nPLAN:\n{task_tree.render()}\nFOCUS TASK: {next_action_id}\n"
                        
                        if str(next_action_id).strip().lower() == "none":
                            dynamic_state += "CRITICAL INSTRUCTION: DO NOT USE TOOLS this turn. Answer the user directly using insights from your THOUGHT.\n"
                            force_final_response = True
                        else:
                            dynamic_state += "CRITICAL INSTRUCTION: Execute ONLY the tool required for the FOCUS TASK. DO NOT HALLUCINATE TOOL OUTPUTS.\n"

                    # -----------------------------------------------------------------
                    # QWEN-AGENT METHODOLOGY: Bypass Native Tools & Use String Prompts
                    # -----------------------------------------------------------------
                    target_tool = locals().get("required_tool", "all")
                    # With the planner disabled, default to final generation (stream directly).
                    # If the model returns tool_calls, the turn loop below will handle them.
                    is_final_generation = force_final_response or target_tool.lower() == "none"

                    # Translate messages to bypass strict API validation and emulate Qwen-Agent
                    req_messages = []
                    for m in messages:
                        if m.get("role") == "tool":
                            # Translate tool results to a user message wrapped in <tool_response>
                            req_messages.append({
                                "role": "user", 
                                "content": f"<tool_response name=\"{m.get('name', 'unknown')}\">\n{m.get('content')}\n</tool_response>"
                            })
                        elif m.get("role") == "assistant":
                            # Ensure the LLM remembers its past tool calls by explicitly rendering them as text
                            ast_content = m.get("content") or ""
                            if m.get("tool_calls"):
                                for tc in m["tool_calls"]:
                                    tc_func = tc.get("function", {})
                                    tc_args = tc_func.get("arguments", "{}")
                                    if isinstance(tc_args, str):
                                        try: tc_args_dict = json.loads(tc_args)
                                        except: tc_args_dict = {}
                                    else:
                                        tc_args_dict = tc_args
                                    
                                    xml_call = f'\n<tool_call>\n<function={tc_func.get("name", "")}>\n'
                                    for k, v in tc_args_dict.items():
                                        xml_call += f'<parameter={k}>\n{str(v)}\n</parameter>\n'
                                    xml_call += "</function>\n</tool_call>\n"
                                    if f'<function name="{tc_func.get("name", "")}">' not in ast_content and "<tool_call>" not in ast_content:
                                        ast_content += xml_call
                            req_messages.append({
                                "role": "assistant",
                                "content": ast_content.strip()
                            })
                        elif m.get("role") == "user":
                            content_val = m.get("content", "")
                            has_vision_node = bool(getattr(self.context.llm_client, 'vision_clients', None))
                            if isinstance(content_val, list):
                                if has_vision_node:
                                    text_parts = []
                                    for item in content_val:
                                        if isinstance(item, dict):
                                            if item.get("type") == "text":
                                                text_parts.append(item.get("text", ""))
                                            elif item.get("type") == "image_url":
                                                text_parts.append("[Image attached and passed to vision node]")
                                    content_val = "\n".join(text_parts)
                                    req_messages.append({"role": "user", "content": content_val})
                                else:
                                    # PRESERVE NATIVE VISION: Let Qwen 3.5 process the image locally
                                    req_messages.append({"role": "user", "content": content_val})
                            else:
                                req_messages.append({"role": "user", "content": content_val})
                        else:
                            req_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

                    # Bundle ALL dynamic context
                    transient_injection = f"{active_persona}{fetched_playbook}{fetched_mem_context}{dynamic_state.strip()}"

                    # Append transient state to the LAST message to preserve KV Cache
                    if req_messages and req_messages[-1]["role"] == "user":
                        req_messages[-1]["content"] += f"\n\n[SYSTEM STATE UPDATE]\n{transient_injection}"
                    else:
                        req_messages.append({"role": "user", "content": f"[SYSTEM STATE UPDATE]\n{transient_injection}"})

                    payload = {
                        "model": model, 
                        "messages": req_messages, 
                        "stream": False, 
                        "temperature": active_temp,
                        "top_p": 0.95,
                        "top_k": 20,
                        "min_p": 0.0,
                        "max_tokens": 8192
                    }
                    # CRITICAL: Do NOT send `tools` or `tool_choice` natively in the payload!
                    
                    pretty_log("LLM Request", f"Turn {turn+1} | Temp {active_temp:.2f}", icon=Icons.LLM_ASK)
                    
                    if is_final_generation and stream_response:
                        payload["stream"] = True
                        # Capture outer variables to prevent NameError when finally block deletes them
                        stream_messages_snapshot = list(messages[-10:])
                        stream_tools_snapshot = list(tools_run_this_turn)
                        stream_thought = thought_content
                        stream_model = model
                        
                        # NEW: Capture accumulated intermediate text (like image tags from previous turns)
                        stream_prefix = final_ai_content.strip() + "\n\n" if final_ai_content.strip() else ""

                        async def stream_wrapper():
                            full_content = ""
                            
                            # NEW: Flush intermediate text to the UI as the first stream chunk
                            if stream_prefix:
                                start_chunk = {
                                    "id": f"chatcmpl-{req_id}", "object": "chat.completion.chunk", "created": created_time,
                                    "model": stream_model, "choices": [{"index": 0, "delta": {"content": stream_prefix}, "finish_reason": None}]
                                }
                                yield f"data: {json.dumps(start_chunk)}\n\n".encode('utf-8')
                                full_content += stream_prefix

                            async for chunk in self.context.llm_client.stream_chat_completion(payload, use_coding=has_coding_intent):
                                self.context.last_activity_time = datetime.datetime.now() # Heartbeat
                                yield chunk
                                try:
                                    chunk_str = chunk.decode("utf-8")
                                    if chunk_str.startswith("data: ") and chunk_str.strip() != "data: [DONE]":
                                        chunk_data = json.loads(chunk_str[6:])   
                                        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                            delta = chunk_data["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                full_content += delta["content"]
                                except Exception as e:
                                    logger.debug(f"Stream chunk decode error: {type(e).__name__}")
                            
                            if self.context.args.smart_memory > 0.0 and last_user_content and not forget_was_called and not last_was_failure:
                                micro_msgs = []
                                for m in [msg for msg in stream_messages_snapshot if msg.get("role") in ["user", "assistant"]][-4:]:
                                    role = m.get("role", "user").upper()
                                    clean_content = re.sub(r'```.*?```', '', str(m.get("content", "")), flags=re.DOTALL)
                                    micro_msgs.append(f"{role}: {clean_content[:500].strip()}")
                                clean_ai = re.sub(r'```.*?```', '', full_content, flags=re.DOTALL)
                                recent_arc = "\\n".join(micro_msgs) + f"\\nAI: {clean_ai[:500].strip()}"
                                if getattr(self.context, 'journal', None):

                                    await asyncio.to_thread(self.context.journal.append, 'smart_memory', {'text': recent_arc, 'model': stream_model})
                                
                            # --- EXTRACT & LOG INTERNAL THINKING (STREAM) ---
                            think_matches = re.findall(r'<think>(.*?)(?:</think>|$)', full_content, flags=re.DOTALL | re.IGNORECASE)
                            for think_text in think_matches:
                                clean_think = think_text.strip()
                                if clean_think:
                                    ui_think = clean_think.replace('\n', ' | ')
                                    logger.info(f"PLANNER MONOLOGUE: {ui_think}")
                                    
                                    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                                    print(f"[INFO ] 💭 {timestamp} - [{req_id}] {'='*15} AGENT INTERNAL THINKING {'='*15}", flush=True)
                                    for line in clean_think.split('\n'):
                                        if line.strip():
                                            print(f"[INFO ] 💭 {timestamp} - [{req_id}] {line.strip()}", flush=True)
                                    print(f"[INFO ] 💭 {timestamp} - [{req_id}] {'='*55}", flush=True)

                            if was_complex_task or execution_failure_count > 0:
                                if not force_stop or "READY TO FINALIZE" in stream_thought.upper():
                                    if getattr(self.context, 'journal', None):

                                        await asyncio.to_thread(self.context.journal.append, 'post_mortem', {'user': last_user_content, 'tools': stream_tools_snapshot, 'ai': full_content, 'model': stream_model})
                                        
                        return stream_wrapper(), created_time, req_id

                    # Ensure msg is always defined in this scope
                    msg = {"role": "assistant", "content": "", "tool_calls": []}
                    try:
                        payload["stream"] = True
                        full_content = ""
                        reasoning_content = ""
                        
                        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                        print(f"[INFO ] 💭 {timestamp} - [{req_id}] {'='*15} AGENT INTERNAL THINKING {'='*15}\n[INFO ] 💭 {timestamp} - [{req_id}] ", end="", flush=True)
                        
                        stop_printing = False

                        async for chunk in self.context.llm_client.stream_chat_completion(payload, use_coding=has_coding_intent):
                            self.context.last_activity_time = datetime.datetime.now() # Heartbeat to prevent Hippocampus from waking up
                            try:
                                chunk_str = chunk.decode("utf-8")
                                if chunk_str.startswith("data: ") and chunk_str.strip() != "data: [DONE]":
                                    chunk_data = json.loads(chunk_str[6:])   
                                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                        delta = chunk_data["choices"][0].get("delta", {})
                                        
                                        if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                                            r_token = delta["reasoning_content"]
                                            reasoning_content += r_token
                                            if not stop_printing:
                                                if "<tool_call" in reasoning_content.lower():
                                                    stop_printing = True
                                                if not stop_printing:
                                                    safe_token = r_token.replace("\n", f"\n[INFO ] 💭 {timestamp} - [{req_id}] ")
                                                    print(safe_token, end="", flush=True)

                                        if "content" in delta and delta["content"] is not None:
                                            text_chunk = delta["content"]
                                            full_content += text_chunk
                                            
                                            if not stop_printing:
                                                if "<tool_call" in full_content.lower():
                                                    stop_printing = True
                                                if not stop_printing and not reasoning_content:
                                                    safe_token = text_chunk.replace("\n", f"\n[INFO ] 💭 {timestamp} - [{req_id}] ")
                                                    print(safe_token, end="", flush=True)
                                                    
                                        if "tool_calls" in delta and delta["tool_calls"]:
                                            if not msg.get("tool_calls"):
                                                msg["tool_calls"] = []
                                            for tc_chunk in delta["tool_calls"]:
                                                idx = tc_chunk.get("index", 0)
                                                while len(msg["tool_calls"]) <= idx:
                                                    msg["tool_calls"].append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                                                
                                                if tc_chunk.get("id"):
                                                    msg["tool_calls"][idx]["id"] = tc_chunk["id"]
                                                if tc_chunk.get("function"):
                                                    fn_chunk = tc_chunk["function"]
                                                    if fn_chunk.get("name"):
                                                        msg["tool_calls"][idx]["function"]["name"] += fn_chunk["name"]
                                                    if fn_chunk.get("arguments"):
                                                        msg["tool_calls"][idx]["function"]["arguments"] += fn_chunk["arguments"]
                            except Exception as e:
                                logger.debug(f"XML Tool parse text stream chunk error: {type(e).__name__}")
                        
                        print(f"\n[INFO ] 💭 {timestamp} - [{req_id}] {'='*55}", flush=True)

                        merged_content = full_content
                        if reasoning_content:
                            merged_content = f"<think>\n{reasoning_content}\n</think>\n" + full_content
                            
                        msg["content"] = merged_content
                    except (httpx.ConnectError, httpx.ConnectTimeout):
                        final_ai_content = "CRITICAL: The upstream LLM server is unreachable. It may have crashed due to memory pressure or is currently restarting. Please wait a moment and try again."
                        pretty_log("System Fault", "Upstream server unreachable", level="ERROR", icon=Icons.FAIL)
                        force_stop = True
                        break
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 400 and "context" in e.response.text.lower():
                            pretty_log("Context Overflow", "Emergency pruning triggered...", icon=Icons.WARN)
                            # Emergency Prune: Keep System + Last User + 1 Last Tool Result (Truncated)
                            system_msgs = [m for m in req_messages if m.get("role") == "system"]
                            last_user = next((m for m in reversed(req_messages) if m.get("role") == "user"), None)
                            
                            recovery_msgs = list(system_msgs)
                            if last_user: recovery_msgs.append(last_user)
                            
                            # If the last thing was a tool output that caused the overflow, keep it but heavily truncated
                            if tools_run_this_turn:
                                last_tool = tools_run_this_turn[-1].copy()
                                last_tool["content"] = last_tool["content"][:1000] + "\n... [EMERGENCY TRUNCATION] ..."
                                recovery_msgs.append(last_tool)
                                
                            recovery_msgs.append({"role": "user", "content": "SYSTEM ALERT: The conversation history was truncated to fit within context limits. Continue task. Assume previous context has been handled."})
                            
                            # RETRY ONCE with pruned context
                            try:
                                payload["messages"] = recovery_msgs
                                messages = recovery_msgs
                                data = await self.context.llm_client.chat_completion(payload, use_coding=has_coding_intent)
                                if "choices" in data and len(data["choices"]) > 0:
                                    msg = data["choices"][0]["message"]
                            except Exception as retry_e:
                                final_ai_content = f"CRITICAL: Context overflow recovery failed: {str(retry_e)}"
                                force_stop = True
                                break
                        else:
                            final_ai_content = f"CRITICAL: Upstream error {e.response.status_code}: {e.response.text}"
                            pretty_log("System Fault", f"HTTP {e.response.status_code}", level="ERROR", icon=Icons.FAIL)
                            force_stop = True
                            break
                    except Exception as e:
                        final_ai_content = f"CRITICAL: An unexpected error occurred while communicating with the LLM: {str(e)}"
                        pretty_log("System Fault", str(e), level="ERROR", icon=Icons.FAIL)
                        force_stop = True
                        break

                    content = msg.get("content") or ""

                    # Merge upstream reasoning_content if present (some models return it as a separate field)
                    if msg.get("reasoning_content"):
                        content = f"<think>\n{msg.get('reasoning_content')}\n</think>\n" + content

                    tool_calls = []
                    ui_content = content
                    
                    # --- EXTRACT & LOG INTERNAL THINKING ---
                    think_matches = re.findall(r'<think>(.*?)(?:</think>|$)', content, flags=re.DOTALL | re.IGNORECASE)
                    for think_text in think_matches:
                        clean_think = think_text.strip()
                        if clean_think:
                            ui_think = clean_think.replace('\n', ' | ')

                            # 1. Trigger UI Planner Monologue Box
                            logger.info(f"PLANNER MONOLOGUE: {ui_think}")

                            # 2. Print multiline thinking safely to the terminal (REMOVED - now handled via streaming)
                            # Timestamp and iterative printing to the console has been moved to the streaming block above.

                    # ---------------------------------------------------------
                    #   ROBUST XML TOOL PARSER
                    # ---------------------------------------------------------
                    # Isolate actual output from <think> blocks FIRST
                    parse_target = re.sub(r'<think>.*?(?:</think>|$)', '', content, flags=re.DOTALL | re.IGNORECASE)
                    
                    has_tool_tag = re.search(r'<tool_call', parse_target, re.IGNORECASE) is not None
                    
                    if has_tool_tag:
                        pretty_log("Agent Parser", "Extracting XML tool call...", icon=Icons.TOOL_CODE)
                        
                        # Split by <tool_call> to handle missing closing tags, spaces, and markdown injections
                        blocks = re.split(r'<tool_call.*?>', parse_target, flags=re.IGNORECASE)
                        for block in blocks[1:]:
                            # Strip out anything after the closing tag if it exists
                            block_content = re.split(r'</tool_call.*?>', block, flags=re.IGNORECASE)[0]
                            
                            try:
                                # Fallback: if it's pure JSON wrapped in <tool_call> without <function>
                                if '<function' not in block_content.lower():
                                    try:
                                        t_data = extract_json_from_text(block_content)
                                        if t_data and "name" in t_data:
                                            # We fake the func_match behavior
                                            func_match = True
                                    except: pass
                                        
                                func_match = re.search(r'<function(?:\s+name=|=)(.*?)>', block_content, re.IGNORECASE)
                                if func_match:
                                    func_name = func_match.group(1).strip().strip('"').strip("'")
                                    args_val = {}
                                    
                                    # Format 1: <parameter name="x">y</parameter>
                                    param_matches = list(re.finditer(r'<parameter(?:\s+name=|=)([^>]+)>(.*?)</parameter>', block_content, re.DOTALL | re.IGNORECASE))
                                    if param_matches:
                                        for p in param_matches:
                                            p_name = p.group(1).strip().strip('"').strip("'")
                                            p_val = p.group(2).strip()
                                            args_val[p_name] = p_val
                                    else:
                                        # Format 2: <parameter name="x" value="y" /> or <parameter name="x" value="y"></parameter>
                                        alt_matches = list(re.finditer(r'<parameter\s+name=["\']([^"\']+)["\']\s+value=["\']([^"\']+)["\']\s*(?:/|>.*?</parameter>)', block_content, re.DOTALL | re.IGNORECASE))
                                        if alt_matches:
                                            for p in alt_matches:
                                                args_val[p.group(1)] = p.group(2)
                                        else:
                                            # Format 3: Bare tags <action>check_health</action>
                                            bare_tags = list(re.finditer(r'<([a-zA-Z0-9_-]+)>(.*?)</\1>', block_content, re.DOTALL | re.IGNORECASE))
                                            # filter out <function> which is already handled
                                            bare_tags = [b for b in bare_tags if b.group(1).lower() != 'function']
                                            if bare_tags:
                                                for b in bare_tags:
                                                    args_val[b.group(1)] = b.group(2).strip()
                                            else:
                                                # Format 4: Attribute tags <parameter action="check_health" />
                                                attr_tags = list(re.finditer(r'<parameter\s+([a-zA-Z0-9_-]+)=["\']([^"\']+)["\']\s*(?:/|>.*?</parameter>)', block_content, re.DOTALL | re.IGNORECASE))
                                                if attr_tags:
                                                    for a in attr_tags:
                                                        if a.group(1).lower() != 'name':  # skip name= as it should have been caught by format 1 or 2
                                                            args_val[a.group(1)] = a.group(2)
                                                else:
                                                    # Format 5: Direct attribute tags <action="check_health">
                                                    direct_attr = list(re.finditer(r'<([a-zA-Z0-9_-]+)=["\']([^"\']+)["\']\s*(?:/|>.*?</\1>|>)', block_content, re.DOTALL | re.IGNORECASE))
                                                    if direct_attr:
                                                        for d in direct_attr:
                                                            args_val[d.group(1)] = d.group(2)
                                            
                                    t_data = {"name": func_name, "arguments": args_val}
                                else:
                                    t_data = extract_json_from_text(block_content)
                                
                                if t_data and "name" in t_data:
                                    # Fix for models that stringify the arguments dict
                                    args_val = t_data.get("arguments", {})
                                    if isinstance(args_val, str):
                                        try: args_val = json.loads(args_val, strict=False)
                                        except: args_val = {}
                                        
                                    tool_calls.append({
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {
                                            "name": t_data.get("name"),
                                            "arguments": json.dumps(args_val)
                                        }
                                    })
                                else:
                                    # Inject parsing failure so the agent knows to retry
                                    tool_calls.append({
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {
                                            "name": "system_parse_error",
                                            "arguments": "{}"
                                        }
                                    })
                            except Exception as e:
                                logger.debug(f"XML execution metadata parsing error: {type(e).__name__}")
                                
                        # Scrub from the human-facing UI string
                        ui_content = re.sub(r'<tool_call.*?>.*?(?:</tool_call.*?>|$)', '', ui_content, flags=re.DOTALL | re.IGNORECASE).strip()
                    else:
                        # Fallback: honour native tool_calls if the model didn't use XML format
                        tool_calls = list(msg.get("tool_calls") or [])
                    
                    ui_content = re.sub(r'<think>.*?(?:</think>|$)', '', ui_content, flags=re.DOTALL | re.IGNORECASE).strip()

                    # --- HALLUCINATION & LEAK SCRUBBERS ---
                    if ui_content:
                        # 1. Hard Truncation for System Prompt Bleed
                        for bleed_marker in ["# Tools", "<tools>", "CRITICAL INSTRUCTION:", "You may call one or more functions", '{"type": "function"']:
                            if bleed_marker in ui_content:
                                ui_content = ui_content.split(bleed_marker)[0]
                                
                        # 2. Regex scrubbers for XML and Execution Artifacts
                        ui_content = re.sub(r'<tool_response>.*?(?:</tool_response>|$)', '', ui_content, flags=re.DOTALL | re.IGNORECASE)
                        ui_content = re.sub(r'--- EXECUTION RESULT ---.*?(?:------------------------|$)', '', ui_content, flags=re.DOTALL)
                        
                        # 3. Task Tree Regurgitation Scrubbers
                        ui_content = re.sub(r'(?m)^\s*(?: )\s*\[.*?\].*?\n?', '', ui_content)
                        ui_content = re.sub(r'(?m)^.*?\((?:IN_PROGRESS|READY|PENDING|DONE|FAILED|BLOCKED)\)\s*\n?', '', ui_content)
                        ui_content = re.sub(r'(?m)^\s*(?:\[)?task_\d+(?:\])?\s*\n?', '', ui_content)
                        ui_content = re.sub(r'(?m)^\s*(?:FOCUS TASK|ACTIVE STRATEGY & PLAN|PLAN|THOUGHT):\s*', '', ui_content)
                        
                        ui_content = ui_content.strip()

                    if ui_content:
                        ui_content = ui_content.replace("\r", "")
                        if final_ai_content and not final_ai_content.endswith("\n\n"):
                            final_ai_content += "\n\n"
                        final_ai_content += ui_content
                        
                    # CRITICAL: Preserve the raw XML tags in the assistant's internal message context so it remembers!
                    msg["content"] = content
                    msg["tool_calls"] = tool_calls
                    
                    if not tool_calls:
                        clean_ui = ui_content.strip("` \n\r")
                        has_img_markdown = bool(re.search(r'!\[.*?\]\(.*?\)', clean_ui))
                        has_valid_image_tool = any(t in raw_tools_called for t in ["image_generation", "execute", "file_system"])
                        has_run_tools = len(tools_run_this_turn) > 0

                        # Catch Stalled Image Mentions
                        if has_img_markdown and not has_valid_image_tool:
                            is_valid_final = "```" in clean_ui or bool(re.search(r'\b(SUCCESS|DONE|COMPLETE)\b', clean_ui.upper()))
                            if not is_valid_final:
                                pretty_log("Agent Parser", "Caught image markdown without tool call.", level="WARNING", icon=Icons.WARN)
                                messages.append({"role": "user", "content": "SYSTEM ALERT: You generated an image markdown tag but didn't call a tool. If you intend to use `vision_analysis` or `image_generation`, output the XML `<tool_call>` block now. If you are finished and just replying to the user, include 'SUCCESS' or 'DONE' in your response."})
                                execution_failure_count += 1
                                continue
                            
                        if not turn_is_conversational and not force_final_response and not is_final_generation:
                            
                            # Catch conversational rambling (allow images to pass, allow long synthesis, allow uncensored disclaimers)
                            is_valid_final = "```" in clean_ui or bool(re.search(r'\b(SUCCESS|DONE|COMPLETE)\b', clean_ui.upper())) or bool(re.search(r'(not legal advice|general information|consult a professional|not medical advice|own risk)', clean_ui.lower()))
                            if len(clean_ui) > 0 and not is_valid_final and not has_img_markdown and not has_run_tools:
                                # We remove the < 300 constraint so aggressive models don't bypass the trap by writing a long essay
                                pretty_log("Agent Parser", "Caught conversational filler instead of tool call", level="WARNING", icon=Icons.WARN)
                                messages.append({"role": "user", "content": "SYSTEM ALERT: You generated a thought process and brief conversational filler, but FAILED to output the actual XML `<tool_call>` block to execute your plan. DO NOT output conversational filler while executing. Output the strict XML `<tool_call>` block now. If you are trying to provide the final answer to the user, you MUST include the word 'SUCCESS' or 'DONE' in your response."})
                                execution_failure_count += 1
                                continue
                                
                        if not clean_ui and not force_final_response and not is_final_generation:
                            pretty_log("Agent Parser", "Model stalled after thinking. Forcing retry.", level="WARNING", icon=Icons.WARN)
                            messages.append({"role": "user", "content": "SYSTEM ALERT: You generated a thought process but stopped abruptly without outputting a valid XML <tool_call> or a response to the user. DO NOT STOP. You must output the required XML <tool_call> block."})
                            execution_failure_count += 1
                            continue
                            
                        user_request_context = last_user_content.lower()
                        has_meta_intent = any(kw in user_request_context for kw in ["learn", "skill", "profile", "lesson", "playbook", "memorize"])
                        meta_tools_called = any(t in raw_tools_called for t in ["learn_skill", "update_profile"])
                        
                        if has_meta_intent and not meta_tools_called and turn < 4:
                            pretty_log("Checklist Nudge", "Enforcing meta-task compliance", icon=Icons.SHIELD)
                            # Remove the recently added content to prevent duplicating text during the loop
                            if content:
                                final_ai_content = final_ai_content[:-len(content)].strip()
                            messages.append({"role": "user", "content": "CRITICAL: You have not fulfilled the learning/profile instructions in the user's request. You MUST call 'learn_skill' or 'update_profile' now before finishing."})
                            continue

                        if self.context.args.smart_memory > 0.0 and last_user_content and not forget_was_called and not last_was_failure:
                            micro_msgs = []
                            for m in [msg for msg in messages if msg.get("role") in ["user", "assistant"]][-4:]:
                                role = m.get("role", "user").upper()
                                clean_content = re.sub(r'```.*?```', '', str(m.get("content", "")), flags=re.DOTALL)
                                micro_msgs.append(f"{role}: {clean_content[:500].strip()}")
                            clean_ai = re.sub(r'```.*?```', '', final_ai_content, flags=re.DOTALL)
                            recent_arc = "\\n".join(micro_msgs) + f"\\nAI: {clean_ai[:500].strip()}"
                            if getattr(self.context, 'journal', None):

                                await asyncio.to_thread(self.context.journal.append, 'smart_memory', {'text': recent_arc, 'model': model})
                        break
                        
                    messages.append(msg)
                    last_was_failure = False
                    redundancy_strikes = 0
                    
                    tool_tasks, tool_call_metadata = [], []
                    for tool in tool_calls:
                        fname = tool["function"]["name"]
                        raw_tools_called.add(fname)
                        tool_usage[fname] = tool_usage.get(fname, 0) + 1
                        

                            
                        if fname == "forget":
                            forget_was_called = True
                        elif fname == "knowledge_base":
                            try:
                                args = json.loads(tool["function"]["arguments"])
                                if args.get("action") == "forget":
                                    forget_was_called = True
                            except: pass

                        max_uses = 3 if fname == "image_generation" else (10 if fname in ["deep_research", "web_search"] else (20 if fname == "execute" else 10))
                        if tool_usage[fname] > max_uses:
                            pretty_log("Loop Breaker", f"Halted overuse: {fname}", icon=Icons.STOP)
                            messages.append({"role": "user", "content": f"SYSTEM ALERT: Tool '{fname}' used too many times in a row. It is now blocked. YOU MUST USE A DIFFERENT APPROACH OR STOP."})
                            force_final_response = True; continue

                        if fname == "system_parse_error":
                            pretty_log("Tool Syntax Error", "Invalid JSON in XML", level="WARNING", icon=Icons.WARN)
                            err_msg = {"role": "tool", "tool_call_id": tool["id"], "name": "system", "content": "SYSTEM ERROR: The JSON inside your <tool_call> was invalid and failed to parse. DO NOT TRY TO FIX THE JSON. Instead, use the native XML format which fully supports raw code without escaping:\\n<tool_call>\\n<function name=\"execute\">\\n<parameter name=\"filename\">script.py</parameter>\\n<parameter name=\"content\">\\n[YOUR UNESCAPED PYTHON CODE HERE]\\n</parameter>\\n</function>\\n</tool_call>"}
                            messages.append(err_msg)
                            tools_run_this_turn.append(err_msg)
                            execution_failure_count += 1
                            last_was_failure = True
                            continue
                            
                        try:
                            t_args = json.loads(tool["function"]["arguments"], strict=False)
                            
                            is_sandbox_mutation = fname in ["execute", "image_generation"] or \
                                                  (fname == "file_system" and t_args.get("operation") in ["write", "replace", "download", "delete", "move", "rename", "unzip", "git_clone"])
                            
                            if is_sandbox_mutation:
                                self.context.cached_sandbox_state = None

                            a_hash = f"{fname}:{json.dumps(t_args, sort_keys=True)}"
                        except Exception as e:
                            err_msg = {"role": "tool", "tool_call_id": tool["id"], "name": fname, "content": f"Error: Invalid JSON arguments - {str(e)}"}
                            messages.append(err_msg)
                            tools_run_this_turn.append(err_msg)
                            execution_failure_count += 1
                            last_was_failure = True
                            continue
                        
                        is_mutating = fname in ["execute", "manage_tasks", "update_profile", "learn_skill", "vision_analysis"] or \
                                      (fname == "file_system" and t_args.get("operation") in ["write", "replace", "download", "delete", "move", "rename"]) or \
                                      (fname == "knowledge_base" and t_args.get("action") in ["ingest_document", "forget", "reset_all", "insert_fact"])

                        if a_hash in seen_tools and not is_mutating and fname != "system_utility":
                            redundancy_strikes += 1
                            pretty_log("Redundancy", f"Blocked duplicate: {fname}", icon=Icons.RETRY)
                            
                            hint = "Change your strategy."
                            if fname == "recall":
                                hint = "Semantic 'recall' cannot do exact string matching. To find an exact line, use file_system 'search'."
                            elif fname == "web_search" or fname == "deep_research":
                                hint = "You MUST change your 'query' text to something completely different or use a different tool."
                            elif fname == "image_generation":
                                hint = "DO NOT call image_generation anymore. Respond directly to the user with the most recent image markdown."
                            elif fname == "file_system":
                                hint = "Check your parameter names carefully. Ensure you are using the correct parameters (e.g., 'destination' for rename, 'content' for write)."
                            else:
                                hint = "You are trapped in a loop. Change your parameters or use a different tool."
                                
                            safe_args_str = json.dumps(t_args)
                            err_msg = {"role": "tool", "tool_call_id": tool["id"], "name": fname, "content": f"TOOL FEEDBACK: Duplicate tool call detected. You just executed `{fname}` with these exact arguments: {safe_args_str}\nRepeating it is redundant and will fail again. {hint}"}
                            messages.append(err_msg)
                            tools_run_this_turn.append(err_msg)
                            if redundancy_strikes >= 3: force_stop = True
                            continue
                            
                        seen_tools.add(a_hash)
                        
                        if fname == "file_system":
                            op = t_args.get("operation")
                            if op == "write":
                                content_val = t_args.get("content")
                                if not content_val or not str(content_val).strip():
                                    pretty_log("Local Guard", "Blocked file_system write with empty content", icon=Icons.STOP)
                                    err_msg = {"role": "tool", "tool_call_id": tool["id"], "name": fname, "content": "SYSTEM BLOCK: You invoked file_system operation='write' but provided an empty or missing 'content' argument. This is completely useless and causes context bloat. Review your task and provide the ACTUAL FULL CONTENT when writing a file. The operation was aborted before execution."}
                                    messages.append(err_msg)
                                    tools_run_this_turn.append(err_msg)
                                    execution_failure_count += 1
                                    last_was_failure = True
                                    continue
                        
                        if fname in self.available_tools:
                            try:
                                tool_tasks.append(self.available_tools[fname](**t_args))
                                tool_call_metadata.append((fname, tool["id"], a_hash, is_mutating))
                            except Exception as e:
                                pretty_log("Tool Invocation Error", str(e), level="WARNING", icon=Icons.WARN)
                                err_msg = {"role": "tool", "tool_call_id": tool["id"], "name": fname, "content": f"Error invoking tool '{fname}' (Did you forget a required argument?): {str(e)}"}
                                messages.append(err_msg)
                                tools_run_this_turn.append(err_msg)
                                last_was_failure = True
                        else: 
                            err_msg = {"role": "tool", "tool_call_id": tool["id"], "name": fname, "content": f"Error: Unknown tool '{fname}'"}
                            messages.append(err_msg)
                            tools_run_this_turn.append(err_msg)
                            execution_failure_count += 1

                    if tool_tasks:
                        results = []
                        for task in tool_tasks:
                            try:
                                res = await task
                            except Exception as e:
                                res = e
                            results.append(res)
                        for i, result in enumerate(results):
                            fname, tool_id, a_hash, is_mutating = tool_call_metadata[i]
                            str_res = str(result).replace("\r", "") if not isinstance(result, Exception) else f"Error: {str(result)}"
                            
                            shield_limit = max(4000, int(char_budget * 0.1))
                            if len(str_res) > shield_limit and fname not in ["file_system", "recall", "deep_research", "web_search", "knowledge_base", "postgres_admin"]:
                                payload = {
                                    "model": model,
                                    "messages": [{"role": "user", "content": f"The user asked: '{last_user_content}'. Summarize this tool output. If it contains facts relevant to the user, extract them. If it is a script error, state the root cause. Output: {str_res[:15000]}"}],
                                    "temperature": 0.0,
                                    "max_tokens": 300
                                }
                                try:
                                    pretty_log("Context Shield", f"Offloading {len(str_res)} chars from {fname} to Edge Worker...", icon=Icons.SHIELD)
                                    summary_data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
                                    summary_content = summary_data["choices"][0]["message"].get("content", "").strip()
                                    if summary_content:
                                        str_res = f"[EDGE CONDENSED]: {summary_content}"
                                except Exception as e:
                                    logger.debug(f"Final generation XML parse failure: {type(e).__name__}")
                                    
                            trunc_limit = max(30000, int(char_budget * 0.4))
                            half = trunc_limit // 2
                            safe_res = str_res[:half] + "\n...[TRUNCATED]...\n" + str_res[-half:] if len(str_res) > trunc_limit else str_res
                            tool_msg = {"role": "tool", "tool_call_id": tool_id, "name": fname, "content": safe_res}
                            messages.append(tool_msg)
                            tools_run_this_turn.append(tool_msg)
                            
                            if fname == "execute":
                                code_match = re.search(r"EXIT CODE:\s*(\d+)", str_res)
                                if code_match:
                                    exit_code_val = int(code_match.group(1))
                                else:
                                    if "Error" in str_res or "Exception" in str_res or "Traceback" in str_res:
                                        exit_code_val = 1
                                    else:
                                        exit_code_val = 0
                                        
                                if exit_code_val != 0:

                                    execution_failure_count += 1

                                    last_was_failure = True
                                    
                                    error_preview = "Unknown Error"
                                    if "STDOUT/STDERR:" in str_res:
                                        error_preview = str_res.split("STDOUT/STDERR:")[1].strip().replace("\n", " ")
                                    elif "SYSTEM ERROR:" in str_res:
                                        error_preview = str_res.split("SYSTEM ERROR:")[1].strip().split("\n")[0]
                                    else:
                                        error_preview = str_res[:60].replace("\n", " ")
                                        
                                    pretty_log("Execution Fail", f"Strike {execution_failure_count}/3 -> {error_preview}", icon=Icons.FAIL)
                                    from ..tools.file_system import tool_list_files
                                    sandbox_state = await tool_list_files(self.context.sandbox_dir, self.context.memory_system)
                                    messages.append({"role": "user", "content": f"AUTO-DIAGNOSTIC: The script failed with an unexpected error. Try a different approach or fix the bug. Execution details: {error_preview}"})
                                    if execution_failure_count == 2:
                                        pretty_log("System 3 Crisis Intervention Triggered", "Engaging meta-cognitive pivot...", icon=Icons.BRAIN_THINK)
                                        sys3_result = await self._run_system_3_pivot(
                                            task_context=last_user_content,
                                            error_context=str_res,
                                            sandbox_state=str(sandbox_state),
                                            model=model
                                        )
                                        if sys3_result.get("tree_update"):
                                            task_tree.load_from_json(sys3_result["tree_update"])
                                            current_plan_json = task_tree.to_json()
                                            execution_failure_count = 0
                                            last_was_failure = False
                                            messages.append({"role": "user", "content": f"SYSTEM 3 PIVOT: The previous approach failed. The strategy has been entirely rewritten. Justification: {sys3_result.get('justification')}. Follow the new plan."})
                                            continue
                                    if execution_failure_count >= 3:
                                        pretty_log("Loop Breaker", "Forcing final response", icon=Icons.STOP)
                                        messages.append({"role": "user", "content": "SYSTEM ALERT: You have failed 3 times in a row. The task cannot be completed. Provide a final response explaining the situation."})
                                        force_final_response = True
                                else:
                                    execution_failure_count = 0
                                    if is_mutating: seen_tools.clear()

                                    pretty_log("Execution Ok", "Script completed with exit code 0", icon=Icons.OK)
                                    # force_stop removed to allow tool chaining
                                        
                            elif str_res.startswith("Error:") or str_res.startswith("Critical Tool Error"):
                                execution_failure_count += 1
                                last_was_failure = True
                                if not force_stop:
                                    error_preview = str_res.replace("Error:", "").strip()
                                    pretty_log("Tool Warning", f"{fname} -> {error_preview}", icon=Icons.WARN)
                                    if execution_failure_count >= 3:
                                        pretty_log("Loop Breaker", "Too many sequential tool failures.", icon=Icons.STOP)
                                        messages.append({"role": "user", "content": "SYSTEM ALERT: You have failed 3 times in a row. Stop trying this approach and try something completely different."})
                                        force_stop = True
                                    
                            elif fname in ["manage_tasks", "learn_skill", "update_profile"] and "SUCCESS" in str_res.upper():
                                # Let the agent naturally answer the user instead of halting abruptly.
                                if is_mutating: seen_tools.clear()
                                pass
                            elif fname == "image_generation" and "SUCCESS" in str_res.upper():
                                # Let the agent naturally answer the user instead of halting abruptly.
                                # DO NOT clear seen_tools for image_generation, so it can't loop the exact same prompt immediately
                                pass

                            else:
                                execution_failure_count = 0
                                if is_mutating: seen_tools.clear()

                # --- FINAL OUTPUT SCRUBBER ---
                # Apply scrubbers FIRST so we don't accidentally scrub our own manual fallback injections
                bleed_markers = [
                    "# Tools", "<tools>", "CRITICAL INSTRUCTION:", "You may call one or more functions", 
                    '{"type": "function"', "SPECIALIST SUBSYSTEM ACTIVATED", "ENGINEERING STANDARDS", 
                    "DYNAMIC SYSTEM STATE", "[SYSTEM STATE UPDATE]"
                ]
                for bleed_marker in bleed_markers:
                    if bleed_marker in final_ai_content:
                        final_ai_content = final_ai_content.split(bleed_marker)[0]
                
                final_ai_content = re.sub(r'<tool_call.*?>.*?(?:</tool_call.*?>|$)', '', final_ai_content, flags=re.DOTALL | re.IGNORECASE)
                final_ai_content = re.sub(r'<tool_response.*?>.*?(?:</tool_response.*?>|$)', '', final_ai_content, flags=re.DOTALL | re.IGNORECASE)
                final_ai_content = re.sub(r'--- EXECUTION RESULT ---.*?(?:------------------------|$)', '', final_ai_content, flags=re.DOTALL)
                final_ai_content = re.sub(r'(?m)^\s*(?:🔄|🟢|⏳|✅|❌|🛑|➖)\s*\[.*?\].*?\n?', '', final_ai_content)
                final_ai_content = re.sub(r'(?m)^.*?\((?:IN_PROGRESS|READY|PENDING|DONE|FAILED|BLOCKED)\)\s*\n?', '', final_ai_content)
                final_ai_content = re.sub(r'(?m)^\s*(?:\[)?task_\d+(?:\])?\s*\n?', '', final_ai_content)
                final_ai_content = re.sub(r'(?m)^\s*(?:FOCUS TASK|ACTIVE STRATEGY & PLAN|PLAN|THOUGHT):\s*', '', final_ai_content)
                final_ai_content = final_ai_content.strip()

                # --- THE "PERFECT IT" PROTOCOL INJECTION ---
                # Only trigger proactive optimization for heavy engineering/research tasks
                heavy_tools_used = any(t.get('name') in ['execute', 'deep_research'] for t in tools_run_this_turn)
                
                if tools_run_this_turn and heavy_tools_used and execution_failure_count == 0 and not last_was_failure and (not final_ai_content or len(final_ai_content) < 50):
                    pretty_log("Perfect It Protocol", "Generating proactive optimization...", icon=Icons.IDEA)
                    perfect_it_prompt = f"Task completed successfully. Final tool output:\n\n{tools_run_this_turn[-1]['content']}\n\n<system_directive>First, succinctly present the tool output/result to the user. Then, based on your Perfection Protocol, analyze the result and proactively suggest one concrete way to optimize, scale, secure, or automate this work further. RESPOND IN PLAIN TEXT ONLY. DO NOT USE TOOLS.</system_directive>"
                    messages.append({"role": "user", "content": perfect_it_prompt})
                    
                    p_req_messages = []
                    for m in messages:
                        if m.get("role") == "tool":
                            p_req_messages.append({"role": "user", "content": f"<tool_response name=\"{m.get('name', 'unknown')}\">\n{m.get('content')}\n</tool_response>"})
                        elif m.get("role") == "assistant":
                            p_req_messages.append({"role": "assistant", "content": m.get("content", "")})
                        else:
                            content_val = m.get("content", "")
                            if isinstance(content_val, list):
                                text_parts = []
                                for item in content_val:
                                    if isinstance(item, dict):
                                        if item.get("type") == "text":
                                            text_parts.append(item.get("text", ""))
                                        elif item.get("type") == "image_url":
                                            if bool(getattr(self.context.llm_client, 'vision_clients', None)):
                                                text_parts.append("[Image attached and passed to vision node]")
                                            else:
                                                text_parts.append(item) # Keep image dict
                                content_val = "\n".join(text_parts) if all(isinstance(x, str) for x in text_parts) else text_parts
                            p_req_messages.append({"role": m.get("role", "user"), "content": content_val})

                    payload["messages"] = p_req_messages

                    # 🔴 CRITICAL FIX: Physically remove tools from payload so it cannot hallucinate a tool call
                    if "tools" in payload: del payload["tools"]
                    if "tool_choice" in payload: del payload["tool_choice"]
                    
                    try:
                        perfection_data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
                        p_msg = perfection_data["choices"][0]["message"].get("content", "")
                        p_msg = re.sub(r'<tool_call>.*?</tool_call>', '', p_msg, flags=re.DOTALL | re.IGNORECASE).strip()
                        
                        # 1. User Display (Conditional on Flag)
                        if getattr(self.context.args, 'perfect_it', False):
                            if final_ai_content:
                                final_ai_content += "\n\n" + p_msg
                            else:
                                final_ai_content = p_msg
                        
                        # 2. Internal Learning (Always)
                        if p_msg and getattr(self.context, 'skill_memory', None):
                            await asyncio.to_thread(
                                self.context.skill_memory.learn_lesson,
                                task=f"Optimization Analysis: {last_user_content[:50]}...",
                                mistake="Sub-optimal pattern identified via Perfection Protocol",
                                solution=p_msg,
                                memory_system=self.context.memory_system
                            )
                            pretty_log("Internal Learning", "Saved optimization strategy to playbook.", icon=Icons.MEM_SAVE)

                    except Exception as e:
                        # Only report failure to user if they expected to see it
                        if getattr(self.context.args, 'perfect_it', False) and not final_ai_content:
                            final_ai_content = "Task finished successfully, but optimization generation failed."
                
                if tools_run_this_turn and not final_ai_content:
                    last_out = tools_run_this_turn[-1].get('content', '')
                    
                    if "![Image]" in last_out:
                        final_ai_content = last_out.strip()
                    else:
                        # Extract just the pure STDOUT so the UI fallback is clean
                        if "STDOUT/STDERR:" in last_out:
                            last_out = last_out.split("STDOUT/STDERR:")[1].strip()
                            if "DIAGNOSTIC HINT" in last_out:
                                last_out = last_out.split("DIAGNOSTIC HINT")[0].strip().strip("-").strip()
                                
                        preview = (last_out[:2000] + '\n...[Truncated]') if len(last_out) > 2000 else last_out
                        final_ai_content = f"Process finished successfully.\n\n### Final Output:\n```text\n{preview}\n```"
                
                if not final_ai_content:
                    final_ai_content = "Task executed successfully."

                # --- AUTOMATED POST-MORTEM (AUTO-LEARNING) ---
                if was_complex_task or execution_failure_count > 0:
                    is_complete_failure = (execution_failure_count >= 3)
                    is_valid_success = (not force_stop or "READY TO FINALIZE" in thought_content.upper())
                    
                    if is_valid_success or is_complete_failure:
                        if getattr(self.context, 'journal', None):

                            await asyncio.to_thread(self.context.journal.append, 'post_mortem', {'user': last_user_content, 'tools': list(tools_run_this_turn), 'ai': final_ai_content, 'model': model})

                body["messages"] = messages
                return final_ai_content, created_time, req_id
                
        finally:
            if 'messages' in locals(): del messages
            if 'tools_run_this_turn' in locals(): del tools_run_this_turn
            if 'sandbox_state' in locals(): del sandbox_state
            if 'data' in locals(): del data
            
            pretty_log("Request Finished", special_marker="END")
            request_id_context.reset(token)

    async def _run_system_3_pivot(self, task_context: str, error_context: str, sandbox_state: str, model: str) -> dict:
        """System 3 Crisis Pivot: generate 3 alternative strategies and pick the safest one."""
        try:
            # Step 1: Generate 3 distinct strategies
            pretty_log("System 3 Generator", "Analysing failure and generating alternative strategies...", icon=Icons.BRAIN_THINK)
            gen_user_msg = (
                f"### TASK CONTEXT:\n{task_context}\n\n"
                f"### ERROR CONTEXT (what failed and why):\n{error_context[:3000]}\n\n"
                f"### CURRENT SANDBOX STATE:\n{sandbox_state[:1500]}"
            )
            gen_payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_3_GENERATION_PROMPT},
                    {"role": "user", "content": gen_user_msg}
                ],
                "temperature": 0.7,
                "response_format": {"type": "json_object"}
            }
            use_swarm = bool(getattr(self.context.llm_client, 'swarm_clients', None))
            gen_data = await self.context.llm_client.chat_completion(gen_payload, use_swarm=use_swarm)
            gen_content = gen_data["choices"][0]["message"].get("content", "")
            strategies_json = extract_json_from_text(gen_content)
            strategies = strategies_json.get("strategies", [])
            if not strategies:
                logger.warning("System 3 Generator returned no strategies.")
                return {}

            # Step 2: Evaluate and pick the safest strategy
            pretty_log("System 3 Evaluator", "Selecting safest recovery path...", icon=Icons.BRAIN_THINK)
            eval_user_msg = (
                f"### PROPOSED STRATEGIES:\n{gen_content}\n\n"
                f"### CURRENT SANDBOX STATE:\n{sandbox_state[:1500]}"
            )
            eval_payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_3_EVALUATOR_PROMPT},
                    {"role": "user", "content": eval_user_msg}
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
            eval_data = await self.context.llm_client.chat_completion(eval_payload, use_swarm=use_swarm)
            eval_content = str(eval_data["choices"][0]["message"].get("content") or "")
            result = extract_json_from_text(eval_content)
            pretty_log("System 3 Complete", f"Winning strategy: {result.get('winning_id', '?')} — {result.get('justification', '')[:120]}", icon=Icons.BRAIN_THINK)
            return result
        except Exception as e:
            logger.error(f"System 3 pivot failed: {e}")
            return {}

            