import asyncio
import os
import shlex
import re
import logging
import uuid
import datetime
import ast
import json
from pathlib import Path
from typing import List
from ..utils.logging import Icons, pretty_log
from ..utils.sanitizer import sanitize_code
from .file_system import _get_safe_path

async def tool_execute(filename: str, content: str = None, sandbox_dir: Path = None, sandbox_manager=None, scrapbook=None, args: List[str] = None, memory_dir: Path = None, stateful: bool = False, **kwargs):
    # --- PARAMETER HALLUCINATION HEALING ---
    filename = filename or kwargs.get("file") or kwargs.get("script") or kwargs.get("name")
    content = content or kwargs.get("code") or kwargs.get("script_content") or kwargs.get("text")

    # --- 🛡️ HIJACK LAYER: CODE SANITIZATION ---
    
    # Helper for consistent error reporting
    def _format_error(msg, hint=None):
        out = f"--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\n{msg}"
        if hint:
            out += f"\n\n--- 💡 DIAGNOSTIC HINT ---\n{hint}\n------------------------"
        return out

    if not filename: return _format_error("Error: filename is required.")
    if not sandbox_dir or not sandbox_manager: return _format_error("Error: Sandbox manager not initialized.")

    # 0. VALIDATION: Ensure we are only executing scripts
    ext = str(filename).split('.')[-1].lower()
    if ext not in ["py", "sh", "js"]:
        pretty_log("Execution Blocked", f"Invalid extension: .{ext}", level="WARNING", icon=Icons.STOP)
        tip = "To save data files or web pages (HTML/CSS/JS), use 'file_system(operation=\"write\", ...)' instead."
        return _format_error(f"SYSTEM ERROR: The 'execute' tool is ONLY for running scripts (.py, .sh, .js).\nSYSTEM TIP: {tip}")

    try:
        host_path = _get_safe_path(sandbox_dir, filename)
    except ValueError as ve:
        return _format_error(str(ve))

    is_new_code = True
    if not content:
        if not host_path.exists():
            return _format_error(f"SYSTEM ERROR: File '{filename}' does not exist. You must provide 'content' to create it.")
        content = await asyncio.to_thread(host_path.read_text)
        is_new_code = False

    # 1. Holistic Sanitization
    content, syntax_error = await asyncio.to_thread(sanitize_code, content, str(filename))
    
    if syntax_error:
        pretty_log("Sanitization Failed", syntax_error, level="WARNING", icon=Icons.BUG)
        
        # HTML/Web guard for when the LLM pastes raw HTML into a Python script
        if "<html" in content.lower() or "body {" in content.lower() or "<div" in content.lower() or "font-family:" in content.lower():
            html_tip = "SYSTEM TIP: It looks like you are trying to write HTML, CSS, or JS intended for the browser. DO NOT use the 'execute' tool to create web pages. Use the 'file_system' tool with operation='write' to save the code directly to a file."
            return _format_error(f"Syntax Error Detected: {syntax_error}\n\n{html_tip}")
            
        # We block execution if syntax is clearly invalid to save a roundtrip
        return _format_error(f"Syntax Error Detected: {syntax_error}\nPlease fix the code and try again.")
        
    # 2. Hard Sandbox Guard against Native Tool Imports
    # The LLM frequently hallucinates that native JSON tools are importable Python modules.
    if ext == "py":
        forbidden_modules = [
            "knowledge_base", "system_utility", "file_system", "manage_tasks", 
            "postgres_admin", "web_search", "fact_check", "deep_research", 
            "vision_analysis", "delegate_to_swarm", "recall", "scratchpad", 
            "learn_skill", "update_profile", "dream_mode", "replan"
        ]
        
        # Check for direct imports or pip installs
        for mod in forbidden_modules:
            # We look for simple patterns: import mod, from mod import, !pip install mod
            if re.search(rf"\bimport\s+{mod}\b", content) or re.search(rf"\bfrom\s+{mod}\s+import\b", content) or re.search(rf"pip\s+install\s+{mod}\b", content):
                pretty_log("Sandbox Guard Invoked", f"Blocked hallucinated import: {mod}", level="WARNING", icon=Icons.SHIELD)
                return _format_error(
                    f"SYSTEM ERROR: FORBIDDEN IMPORT DETECTED -> '{mod}'\n"
                    f"CRITICAL: '{mod}' is a Native JSON Tool, NOT a Python module.\n"
                    f"You CANNOT import it or install it in this sandbox.\n"
                    f"To use '{mod}', you MUST stop writing code and call the JSON tool directly!"
                )

    # 3. Final Trim
    clean_content = content.strip()
    
    # Extract rel_path early for wrapper filename generation
    if not filename: return _format_error("Error: filename is required.")
    rel_path = str(filename).lstrip("/")
    if rel_path.startswith("sandbox/"):
        rel_path = rel_path[8:]

    # 4. Stateful Execution Wrapper (Jupyter-like behavior)
    exec_content = clean_content
    exec_rel_path = rel_path
    wrapper_line_offset = 0

    if ext == "py" and stateful:
        pretty_log("Stateful Execution", "Injecting Dill State Manager", icon=Icons.TOOL_CODE)
        state_file = "/workspace/.ghost_state.dill"
        wrapper_prefix = f"""import os, sys
try:
    import dill
except ImportError:
    print("SYSTEM ERROR: 'dill' library missing. Run `pip install dill`.")
    sys.exit(1)

# 1. Load previous state
if os.path.exists("{state_file}"):
    try:
        dill.load_session("{state_file}")
    except Exception as e:
        print(f"Warning: Could not load previous state: {{e}}")

# --- AGENT CODE START ---
"""
        wrapper_suffix = f"""
# --- AGENT CODE END ---

# 2. Save new state
try:
    dill.dump_session("{state_file}")
except Exception as e:
    print(f"\\nWarning: Could not save state for next turn: {{e}}")
"""
        exec_content = wrapper_prefix + clean_content + wrapper_suffix
        exec_rel_path = str(Path(rel_path).parent / f"._{Path(rel_path).name}")
        wrapper_line_offset = wrapper_prefix.count('\n')

    # ----------------------------------------
    if stateful:
        pretty_log("Execution Task", f"{filename} [STATEFUL]", icon=Icons.TOOL_CODE)
    else:
        pretty_log("Execution Task", filename, icon=Icons.TOOL_CODE)
        
    try:
        exec_host_path = _get_safe_path(sandbox_dir, exec_rel_path)
    except ValueError as ve:
        return _format_error(str(ve))
        
    # Stubbornness Guard checks the ORIGINAL file
    if is_new_code and host_path.exists():
        if host_path.stat().st_size < 1_000_000:
            try:
                def _check_same():
                    return "".join(host_path.read_text().split()) == "".join(clean_content.split())
                is_same = await asyncio.to_thread(_check_same)
                if is_same:
                    is_new_code = False
            except: pass
        
    # Async Directory Creation
    if is_new_code:
        await asyncio.to_thread(host_path.parent.mkdir, parents=True, exist_ok=True)
        try: 
            await asyncio.to_thread(host_path.write_text, clean_content)
        except Exception as e: 
            return _format_error(f"Error writing script: {e}")
            
    if stateful and ext == "py":
        try:
            await asyncio.to_thread(exec_host_path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(exec_host_path.write_text, exec_content)
        except Exception as e: 
            return _format_error(f"Error writing stateful wrapper: {e}")

    if rel_path.endswith(".py"):
        await asyncio.to_thread(sandbox_manager.execute, f"python3 -m black {rel_path}", timeout=15)

    try:
        ext = rel_path.split('.')[-1].lower()
        runtime_map = {"py": "python3 -u", "js": "node", "sh": "bash"}
        runner = runtime_map.get(ext, "")
        cmd = f"{runner} {exec_rel_path}" if runner else f"./{exec_rel_path}"
        if args: 
            # SECURITY FIX: Use shlex.quote to safely escape all arguments
            cmd += " " + " ".join(shlex.quote(str(a)) for a in args)

        output, exit_code = await asyncio.to_thread(sandbox_manager.execute, cmd)
        
        diagnostic_info = ""
        if exit_code != 0:
            if stateful and ext == "py":
                # Hide the temporary execution filename from the output
                output = output.replace(exec_rel_path, rel_path)
            
            tb_match = re.findall(r'File "([^"]+)", line (\d+),', output)
            if tb_match:
                # Prioritize matches from the actual script or workspace, ignore deep library traces
                script_matches = [m for m in tb_match if filename in m[0] or rel_path in m[0] or "/workspace/" in m[0] or m[0].startswith("./")]
                
                if script_matches:
                    _, last_error_line = script_matches[-1]
                else:
                    _, last_error_line = tb_match[-1]
                    
                try:
                    line_num = int(last_error_line)
                    if stateful and ext == "py" and (rel_path in output or filename in output):
                         line_num = max(1, line_num - wrapper_line_offset)
                    
                    lines = clean_content.splitlines()
                    start_l = max(0, line_num - 3)
                    end_l = min(len(lines), line_num + 2)
                    snippet = "\n".join([f"{i+1}: {l}" for i, l in enumerate(lines) if start_l <= i < end_l])
                    diagnostic_info = f"Error detected at Line {line_num}:\n{snippet}\n\nSUGGESTION: Review the snippet above line {line_num}."
                except: pass

        if exit_code != 0:
             return _format_error(output, hint=diagnostic_info)
        
        return f"--- EXECUTION RESULT ---\nEXIT CODE: {exit_code}\nSTDOUT/STDERR:\n{output}"
    except Exception as e:
        return _format_error(f"Error: {e}")
    finally:
        if stateful and ext == "py":
            try:
                await asyncio.to_thread(exec_host_path.unlink, missing_ok=True)
            except: pass
