# src/ghost_agent/core/dream.py

import json
import logging
import asyncio
from typing import List, Dict, Any

from .agent import extract_json_from_text
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

class Dreamer:
    """
    Active Memory Consolidation System.
    "Dreams" about recent memories to synthesize them into higher-order facts and extract heuristics.
    """
    def __init__(self, agent_context):
        self.context = agent_context
        self.memory = agent_context.memory_system

    async def dream(self, model_name: str = "qwen-3.5-9b"):
        if not self.memory or not self.memory.collection:
            return "Memory system not available."
            
        pretty_log("Dream Mode", "Entering REM cycle (Consolidating Memory & Extracting Heuristics)...", icon="💤")
        
        try:
            results = await asyncio.to_thread(
                self.memory.collection.get,
                where={"type": "auto"},
                limit=100,
                include=["documents", "metadatas", "embeddings"]
            )
        except Exception as e:
            return f"Dream error: {e}"
            
        ids = results['ids']
        documents = results['documents']
        
        if len(documents) < 3:
            return "Not enough entropy to dream. (Need > 3 auto-memories to form heuristics)"
            
        mem_list = [f"ID:{i} | {doc}" for i, doc in zip(ids, documents)]
        mem_block = "\n".join(mem_list[:50])
        pretty_log("Dream Mode", f"Analyzing {len(ids)} fragments for meta-patterns...", icon="🧠")
        
        prompt = f"""### IDENTITY
You are the Active Memory Consolidation (Dream) Subsystem.

### TASK
Below is a list of raw, fragmented memories from the Ghost Agent's recent tasks.
Your job is twofold:
1. MERGE overlapping facts into single, high-density facts.
2. EXTRACT HEURISTICS: Identify repeating errors or user preferences and translate them into a persistent behavioral rule (e.g., "Always use absolute paths in Docker").

### RAW MEMORIES
{mem_block}

### OUTPUT FORMAT
Return ONLY valid JSON. If no patterns exist, return empty lists.
{{
  "consolidations": [
    {{
      "synthesis": "The user is working on a Python-based Ghost Agent.",
      "merged_ids": ["ID:...", "ID:..."]
    }}
  ],
  "heuristics": [
    "Always wrap Docker network calls in a try/except."
  ]
}}
"""

        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "system", "content": "You are a Memory Optimizer."}, {"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 1024,
            }
            data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
            content = data["choices"][0]["message"]["content"]
            result = extract_json_from_text(content)
            
            consolidations = result.get("consolidations", [])
            heuristics = result.get("heuristics", [])
            
            if not consolidations and not heuristics:
                return "Dream cycle complete. No patterns or heuristics found."
                
            ops_log = []
            
            # Process Merged Facts
            for item in consolidations:
                synthesis = item.get("synthesis")
                merged_ids = item.get("merged_ids", [])
                stripped_ids = [mid.replace("ID:", "").strip() for mid in merged_ids]
                
                if synthesis and len(stripped_ids) > 1:
                    # ADD new fact
                    await asyncio.to_thread(self.memory.add, synthesis, {"type": "consolidated_fact", "timestamp": "DREAM_CYCLE"})
                    # DELETE old fragments
                    await asyncio.to_thread(self.memory.collection.delete, ids=stripped_ids)
                    ops_log.append(f"Merged {len(stripped_ids)} items -> '{synthesis[:50]}...'")
                    pretty_log("Dream Merge", f"Consolidated {len(stripped_ids)} into 1: {synthesis[:40]}...", icon="✨")

            # Process Heuristics (Save to Skills Playbook)
            if heuristics and self.context.skill_memory:
                for h in heuristics:
                    await asyncio.to_thread(
                        self.context.skill_memory.learn_lesson,
                        task="Dream Cycle Heuristic Extraction",
                        mistake="Inefficient or sub-optimal execution patterns.",
                        solution=h,
                        memory_system=self.memory
                    )
                    ops_log.append(f"Learned Heuristic: '{h[:50]}...'")
                    pretty_log("Dream Heuristic", f"Extracted Rule: {h[:40]}...", icon="💡")
                    
            summary = "\n".join(ops_log)
            pretty_log("Dream Wake", f"Consolidation Complete:\n{summary}", icon="☀️")
            
            # Step 3: Compress Playbook
            compress_msg = await self.compress_playbook(model_name)
            if compress_msg:
                summary += f"\nPlaybook Compression: {compress_msg}"

            return f"Dream Complete. Operations:\n{summary}"
            
        except Exception as e:
            return f"Dream failed: {e}"

    async def compress_playbook(self, model_name: str) -> str:
        if not self.context.skill_memory:
            return ""

        try:
            content = await asyncio.to_thread(self.context.skill_memory.file_path.read_text)
            playbook = json.loads(content) if content else []
        except Exception:
            return "Failed to read playbook."

        if len(playbook) < 20:
            return ""

        prompt = "You are a Playbook Compression Engine. Review this JSON array of learned lessons. Merge duplicates, combine related rules into broader heuristics, and remove obsolete ones. Preserve all critical technical constraints. Return ONLY a JSON object with a 'compressed_playbook' array. Schema must remain: [{'task': '', 'mistake': '', 'solution': '', 'timestamp': ''}]."

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(playbook)}
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"}
        }

        try:
            data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
            result = extract_json_from_text(data["choices"][0]["message"]["content"])
            compressed_playbook = result.get("compressed_playbook", [])

            if compressed_playbook and isinstance(compressed_playbook, list):
                await asyncio.to_thread(self.context.skill_memory.save_playbook, compressed_playbook)
                msg = f"Compressed {len(playbook)} rules down to {len(compressed_playbook)}"
                pretty_log("Playbook Compression", msg, icon="🗜️")
                return msg
            return "Compression returned invalid format."
        except Exception as e:
            return f"Compression failed: {e}"

    async def synthetic_self_play(self, model_name: str = "default"):
        from .prompts import SYNTHETIC_CHALLENGE_PROMPT
        from .agent import GhostAgent, extract_json_from_text
        from ..sandbox.docker import DockerSandbox
        from ..utils.logging import Icons
        import tempfile
        import copy
        import asyncio
        from pathlib import Path

        pretty_log("Dream Mode 2.0", "Initiating Synthetic Self-Play sequence...", icon=Icons.BRAIN_THINK)

        # Retrieve targeted weaknesses to generate a localized curriculum
        recent_failures = ""
        if self.context.skill_memory:
            recent_failures = await asyncio.to_thread(self.context.skill_memory.get_recent_failures)

        system_message = SYNTHETIC_CHALLENGE_PROMPT
        if recent_failures:
            system_message += f"\n\n### TARGETED WEAKNESSES\nThe agent recently struggled with these mistakes:\n{recent_failures}\n\nDesign the challenge to explicitly test and train the agent on these specific weaknesses."
            
        # Add strict constraint to prevent token overflow
        system_message += "\n\nCRITICAL: You must keep the `setup_script` extremely concise (under 20 lines). Mock only 2-3 items of data. Output `challenge_prompt` and `validation_script` FIRST in your JSON."
        
        # 1. Generate the challenge
        payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": "You are an AI training coordinator."}, {"role": "user", "content": system_message}],
            "temperature": 0.6,
            "max_tokens": 8192,
        }
        try:
            data = await self.context.llm_client.chat_completion(payload, use_worker=True, is_background=True)
            content_text = data["choices"][0]["message"]["content"]
            result_json = extract_json_from_text(content_text)
            
            challenge = result_json.get("challenge_prompt")
            validation_script = result_json.get("validation_script", "")
            setup_script = result_json.get("setup_script", "")
            
            # Fallback regex extraction if JSON is malformed (e.g. unescaped nested quotes)
            # Fallback regex extraction if JSON is malformed (e.g. unescaped nested quotes or truncated)
            if not challenge or not validation_script:
                import re
                # Removed the strict ending requirement to tolerate truncated strings
                cp_match = re.search(r'"challenge_prompt"\s*:\s*"(.*?)"(?:\s*,|\s*}|$)', content_text, re.DOTALL)
                vs_match = re.search(r'"validation_script"\s*:\s*"(.*?)"(?:\s*,|\s*}|$)', content_text, re.DOTALL)
                ss_match = re.search(r'"setup_script"\s*:\s*"(.*?)"(?:\s*,|\s*}|$)', content_text, re.DOTALL)
                                
                if cp_match: challenge = cp_match.group(1).encode().decode('unicode_escape')
                if vs_match: validation_script = vs_match.group(1).encode().decode('unicode_escape')
                if ss_match: setup_script = ss_match.group(1).encode().decode('unicode_escape')
            
            from ..utils.sanitizer import extract_code_from_markdown
            if validation_script:
                validation_script = extract_code_from_markdown(validation_script)
                
            if setup_script:
                setup_script = extract_code_from_markdown(setup_script)
                
        except Exception as e:
            return f"Failed to generate challenge: {e}"

        if not challenge or not validation_script:
            pretty_log("Self-Play Error", f"Failed to extract challenge and validation script from LLM output:\n{content_text[:500]}...", level="ERROR", icon=Icons.FAIL)
            return "Failed to extract challenge and validation script."
        pretty_log("Synthetic Challenge", challenge[:80] + "...", icon=Icons.TOOL_CODE)

        class ReadOnlySkillMemory:
            def __init__(self, real_sm):
                self.real_sm = real_sm

            def get_playbook_context(self, *args, **kwargs):
                if self.real_sm:
                    return self.real_sm.get_playbook_context(*args, **kwargs)
                return ""

            def learn_lesson(self, *args, **kwargs):
                pass

            def save_playbook(self, *args, **kwargs):
                pass

            def __getattr__(self, name):
                if self.real_sm:
                    return getattr(self.real_sm, name)
                raise AttributeError(name)

        class ReadOnlyVectorMemory:
            def __init__(self, real_vm):
                self.real_vm = real_vm

            def search(self, *args, **kwargs):
                if self.real_vm:
                    return self.real_vm.search(*args, **kwargs)
                return []

            def search_advanced(self, *args, **kwargs):
                if self.real_vm:
                    return self.real_vm.search_advanced(*args, **kwargs)
                return []

            def add(self, *args, **kwargs):
                pass

            def smart_update(self, *args, **kwargs):
                pass

            def delete(self, *args, **kwargs):
                pass

            def __getattr__(self, name):
                if self.real_vm:
                    return getattr(self.real_vm, name)
                raise AttributeError(name)

        # 2. Setup an isolated, temporary context so we don't pollute the user's real workspace
        with tempfile.TemporaryDirectory() as temp_sandbox:
            isolated_context = copy.copy(self.context)
            isolated_context.sandbox_dir = Path(temp_sandbox)
            isolated_context.args = copy.copy(self.context.args)
            isolated_context.args.perfect_it = False
            isolated_context.args.smart_memory = 0.0
            isolated_context.profile_memory = None
            isolated_context.scheduler = None
            isolated_context.memory_system = ReadOnlyVectorMemory(self.context.memory_system)
            isolated_context.skill_memory = ReadOnlySkillMemory(self.context.skill_memory)
            
            from ..memory.scratchpad import Scratchpad
            isolated_context.scratchpad = Scratchpad()

            isolated_context.sandbox_manager = DockerSandbox(isolated_context.sandbox_dir, isolated_context.tor_proxy)

            try:
                validator_path = Path(temp_sandbox) / ".validator.py"
                await asyncio.to_thread(validator_path.write_text, validation_script)

                await asyncio.to_thread(isolated_context.sandbox_manager.ensure_running)
                temp_agent = GhostAgent(isolated_context)
                temp_agent.available_tools.pop("self_play", None)
                for tool_name in ["manage_tasks", "postgres_admin", "update_profile", "learn_skill", "delegate_to_swarm", "system_utility"]:
                    temp_agent.available_tools.pop(tool_name, None)

                body = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": f"### SYNTHETIC TRAINING EXERCISE\nSolve this challenge perfectly.\n\n{challenge}"}]
                }

                passed = False
                for attempt in range(5):
                    pretty_log("Self-Play", f"Commencing Attempt {attempt + 1}/5", icon=Icons.TOOL_CODE)
                    final_ai_content, _, _ = await temp_agent.handle_chat(body, background_tasks=None)
                    
                    # Early abort if the agent gets hopelessly stuck or blows out context
                    if "SYSTEM ALERT: You have failed 3 times" in final_ai_content or "CRITICAL:" in final_ai_content:
                        pretty_log("Self-Play Abort", "Agent hit a hard failure state. Aborting loop early.", level="WARNING", icon=Icons.STOP)
                        break
                    
                    # Extract the actual tool execution outputs so the judge isn't blind
                    execution_trace = temp_agent._get_recent_transcript(body["messages"])

                    try:
                        output, exit_code = await asyncio.to_thread(isolated_context.sandbox_manager.execute, "python3 .validator.py")
                        passed = (exit_code == 0)
                        
                        if passed:
                            pretty_log("Self-Play", "Tests Passed: Challenge Solved", icon=Icons.OK)
                            break
                        else:
                            feedback = str(output).strip() if output else "Validation script failed silently (no output)."
                            
                            # Circuit breaker for broken validator scripts
                            if "SyntaxError" in feedback and ".validator.py" in feedback:
                                pretty_log("Self-Play Abort", "Validator script has syntax errors. Aborting.", level="ERROR", icon=Icons.STOP)
                                break
                                
                            if len(feedback) > 1500:
                                feedback = feedback[:1500] + "\n...[TRUNCATED FOR LENGTH]"
                                
                            pretty_log("Self-Play Judge Rejection", feedback[:500].replace('\n', ' ') + "...", level="WARNING", icon=Icons.FAIL)
                            body["messages"].append({"role": "user", "content": f"SYSTEM JUDGE REJECTION: You did not solve the task. Feedback:\n{feedback}\nYou must fix the code and try again."})
                    except Exception as e:
                        pretty_log("Self-Play Judge", f"Test execution failed: {e}", level="WARNING", icon=Icons.FAIL)
                        break

                
                # --- GENUINE LEARNING EXTRACTION ---
                if passed:
                    status_str = f"SUCCESS (in {attempt + 1} attempts)"
                elif attempt < 4:
                    status_str = f"FAILURE (Aborted on attempt {attempt + 1})"
                else:
                    status_str = "FAILURE (Exhausted 5 attempts)"
                    
                pretty_log("Self-Play Analysis", "Extracting genuine lessons from simulation...", icon=Icons.BRAIN_THINK)
                
                # We use the REAL context to save the lesson, jumping out of the isolated simulation
                if self.context.skill_memory:
                    transcript = temp_agent._get_recent_transcript(body["messages"])
                    if len(transcript) > 6000:
                        transcript = "...[EARLIER ATTEMPTS TRUNCATED]...\n" + transcript[-6000:]
                        
                    if attempt == 0 and passed:
                        learn_prompt = f"### SELF-PLAY POST-MORTEM\nThe agent effortlessly solved a simulated challenge on the first try.\n\nCHALLENGE:\n{challenge}\n\nTRANSCRIPT:\n{transcript}\n\nIdentify a 'Best Practice' or 'Optimization' used by the agent. Extract a concrete rule to enforce this in the future. Return ONLY a JSON object with 'task', 'mistake' (leave empty), and 'solution' (the best practice)."
                    elif passed:
                        learn_prompt = f"### SELF-PLAY POST-MORTEM\nThe agent initially struggled, but eventually overcame the friction and SUCCEEDED.\n\nCHALLENGE:\n{challenge}\n\nTRANSCRIPT:\n{transcript}\n\nIdentify the exact mistake the agent made initially, and the specific correction that led to success. Return ONLY a JSON object with 'task', 'mistake' (what it did wrong first), and 'solution' (how it fixed it)."
                    else:
                        learn_prompt = f"### SELF-PLAY POST-MORTEM\nThe agent attempted a simulated challenge and FAILED completely.\n\nCHALLENGE:\n{challenge}\n\nTRANSCRIPT:\n{transcript}\n\nIdentify the core technical error or strategy flaw. Extract a concrete rule to prevent this in the future. Return ONLY a JSON object with 'task', 'mistake', and 'solution' (what it should have done instead)."
                    
                    try:
                        learn_payload = {"model": model_name, "messages": [{"role": "system", "content": "You are a Meta-Cognitive Analyst. Output JSON."}, {"role": "user", "content": learn_prompt}], "temperature": 0.1}
                        l_data = await self.context.llm_client.chat_completion(learn_payload, use_worker=True)
                        l_json = extract_json_from_text(l_data["choices"][0]["message"].get("content", ""))
                        
                        if all(k in l_json for k in ["task", "mistake", "solution"]):
                            await asyncio.to_thread(
                                self.context.skill_memory.learn_lesson, 
                                f"[Self-Play] {l_json['task']}", 
                                l_json['mistake'], 
                                l_json['solution'], 
                                memory_system=self.context.memory_system
                            )
                            
                            if self.context.scratchpad:
                                report = f"Challenge: {challenge}\nStatus: {status_str}\nLearned task: {l_json['task']}\nMistake: {l_json['mistake']}\nSolution: {l_json['solution']}"
                                self.context.scratchpad.set("Self-Play Report", report)
                                
                    except Exception as e:
                        logger.error(f"Self-play learning extraction failed: {e}")
                        if self.context.scratchpad:
                            self.context.scratchpad.set("Self-Play Report", f"Challenge: {challenge}\nStatus: {status_str}\nExtraction error: {e}")
                else:
                    if self.context.scratchpad:
                        self.context.scratchpad.set("Self-Play Report", f"Challenge: {challenge}\nStatus: {status_str}\n(No skills extraction configured)")

                pretty_log("Self-Play Concluded", f"Simulation ended with status: {status_str}.", icon=Icons.OK)
                
            except Exception as e:
                pretty_log("Self-Play Error", str(e), level="ERROR", icon=Icons.FAIL)
                return f"Self-Play encountered an error: {e}"
            finally:
                if isolated_context.sandbox_manager and isolated_context.sandbox_manager.container:
                    try:
                        isolated_context.sandbox_manager.container.remove(force=True)
                    except: pass
                    
        report_str = "A detailed post-mortem analysis has been saved to your scratchpad."
        if hasattr(self.context, 'scratchpad') and self.context.scratchpad:
            report_val = self.context.scratchpad.get("Self-Play Report")
            if report_val:
                report_str = f"SELF-PLAY POST-MORTEM REPORT:\n{report_val}"
                
        return f"Synthetic Self-Play cycle completed. Final Status: {status_str}.\n\n{report_str}"