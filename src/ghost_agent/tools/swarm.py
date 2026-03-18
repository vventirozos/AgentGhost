import asyncio
from ..utils.logging import Icons, pretty_log

# Keep strong references to background tasks to prevent aggressive garbage collection
_swarm_tasks = set()

async def _swarm_worker(instruction: str, input_data: str, output_key: str, llm_client, fallback_model_name: str, scratchpad, target_model: str = None):
    """Background worker that executes strictly on the fast edge node."""
    node = llm_client.get_swarm_node(target_model)
    if not node:
        scratchpad.set(output_key, "SYSTEM ALERT: Swarm execution failed. No cluster nodes available.")
        return
        
    client = node["client"]
    model_name = node["model"]

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a specialized Swarm Worker node. Execute the user's instruction on the provided data and return ONLY the results. Be concise."},
            {"role": "user", "content": f"INSTRUCTION:\n{instruction}\n\nINPUT DATA:\n{input_data[:20000]}"}
        ],
        "temperature": 0.0,
        "max_tokens": 2048
    }

    try:
        # 1. We bypass `chat_completion` fallback to AVOID blocking the Mac Mini's local queue.
        resp = await client.post("/v1/chat/completions", json=payload, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        result_text = data["choices"][0]["message"].get("content", "").strip()
        
        scratchpad.set(output_key, result_text)
        pretty_log("Swarm Task", f"Completed '{output_key}' on node {model_name}", icon=Icons.OK)
        
    except Exception as e:
        pretty_log("Swarm Task Failed", f"Edge node offline: {e}", level="WARNING", icon=Icons.WARN)
        scratchpad.set(output_key, f"SYSTEM ALERT: Swarm execution failed ({e}). The edge node is offline. You must process this data yourself synchronously.")

async def tool_delegate_to_swarm(llm_client, model_name: str, scratchpad, tasks: list = None, instruction: str = None, input_data: str = None, output_key: str = None, **kwargs):
    """Dispatches tasks to the background swarm and immediately returns."""
    if not scratchpad:
        return "Error: Scratchpad memory is not initialized."

    if getattr(llm_client, 'swarm_clients', None) is None or len(llm_client.swarm_clients) == 0:
        return "SYSTEM WARNING: The Swarm Cluster is not configured (no --swarm-nodes provided). Do not use this tool anymore. You MUST process this task synchronously in your main context."
        
    if tasks is None:
        tasks = []
        
    # Backwards compatibility
    if instruction and input_data and output_key:
        tasks.append({
            "instruction": instruction,
            "input_data": input_data,
            "output_key": output_key
        })
        
    if not tasks:
        return "Error: No tasks provided to delegate_to_swarm."
    
    pretty_log("Swarm Dispatch", f"Delegating {len(tasks)} tasks to cluster", icon=Icons.BRAIN_PLAN)
    
    for task_def in tasks:
        t_instruction = task_def.get("instruction")
        t_input_data = task_def.get("input_data")
        t_output_key = task_def.get("output_key")
        t_target_model = task_def.get("target_model")
        
        if not t_instruction or not t_input_data or not t_output_key:
            pretty_log("Swarm Skip", f"Skipping invalid task definition: {task_def}", level="WARNING", icon=Icons.WARN)
            continue
            
        task = asyncio.create_task(_swarm_worker(t_instruction, t_input_data, t_output_key, llm_client, model_name, scratchpad, target_model=t_target_model))
        _swarm_tasks.add(task)
        task.add_done_callback(_swarm_tasks.discard)

    return f"SUCCESS: {len(tasks)} task(s) dispatched to the Swarm. The results will be silently written to your SCRAPBOOK when finished. Do not wait—continue executing your next planned steps immediately."
