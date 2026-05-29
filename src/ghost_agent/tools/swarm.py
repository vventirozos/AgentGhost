import asyncio
import uuid
from ..utils.logging import Icons, pretty_log

# Keep strong references to background tasks to prevent aggressive garbage collection
_swarm_tasks = set()

# Module-level registry of dispatched fire-and-forget tasks keyed by task id
# so the agent can poll status from another turn. Bounded loosely — old
# completed entries fall off via the done-callback.
_swarm_task_registry: dict[str, asyncio.Task] = {}


def _register_task(task_id: str, task: asyncio.Task):
    _swarm_task_registry[task_id] = task

    def _cleanup(_t, _id=task_id):
        _swarm_task_registry.pop(_id, None)

    task.add_done_callback(_cleanup)

async def _swarm_worker(instruction: str, input_data: str, output_key: str, llm_client, fallback_model_name: str, scratchpad, worker_persona: str = None, target_model: str = None, preselected_node=None):
    """Background worker that executes on the fast edge node with retry logic."""
    MAX_RETRIES = 2

    sys_prompt = worker_persona if worker_persona else "You are a specialized Swarm Worker node. Execute the user's instruction on the provided data and return ONLY the results. Be concise."

    payload = {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"INSTRUCTION:\n{instruction}\n\nINPUT DATA:\n{input_data[:20000]}"}
        ],
        "temperature": 0.0,
        "max_tokens": 2048
    }

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        # On the first attempt reuse the node the dispatcher already
        # resolved + validated for this task. Re-resolving here advanced
        # the round-robin index a SECOND time per task — skewing node
        # assignment and (for an unset target_model) sending the worker to
        # a different node than the one validated. Retries DO re-resolve so
        # a transient failure can route elsewhere.
        if attempt == 0 and preselected_node is not None:
            node = preselected_node
        else:
            node = llm_client.get_swarm_node(target_model)
        if not node:
            scratchpad.set(output_key, "SYSTEM ALERT: Swarm execution failed. No cluster nodes available.")
            return

        client = node["client"]
        model_name = node["model"]
        payload["model"] = model_name

        try:
            resp = await client.post("/v1/chat/completions", json=payload, timeout=300.0)
            resp.raise_for_status()
            data = resp.json()
            result_text = data["choices"][0]["message"].get("content", "").strip()

            # Record success with circuit breaker if available
            cb = getattr(llm_client, 'circuit_breaker', None)
            if cb and node.get("url"):
                cb.record_success(node["url"])

            scratchpad.set(output_key, result_text)
            if attempt > 0:
                pretty_log("Swarm Task", f"Completed '{output_key}' on node {model_name} (after {attempt} retries)", icon=Icons.OK)
            else:
                pretty_log("Swarm Task", f"Completed '{output_key}' on node {model_name}", icon=Icons.OK)
            return

        except Exception as e:
            last_error = e
            # Record failure with circuit breaker
            cb = getattr(llm_client, 'circuit_breaker', None)
            if cb and node.get("url"):
                cb.record_failure(node["url"])

            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                pretty_log("Swarm Retry", f"Attempt {attempt + 1} failed ({type(e).__name__}), retrying in {wait}s...", level="WARNING", icon=Icons.RETRY)
                await asyncio.sleep(wait)
            else:
                pretty_log("Swarm Task Failed", f"All {MAX_RETRIES + 1} attempts failed: {e}", level="WARNING", icon=Icons.WARN)

    scratchpad.set(output_key, f"SYSTEM ALERT: Swarm execution failed after {MAX_RETRIES + 1} attempts ({last_error}). The edge node is offline. You must process this data yourself synchronously.")

async def tool_delegate_to_swarm(llm_client, model_name: str, scratchpad, tasks: list = None, instruction: str = None, input_data: str = None, output_key: str = None, worker_persona: str = None, await_results: bool = False, **kwargs):
    """Dispatch tasks to the background swarm.

    By default (``await_results=False``) this is fire-and-forget: each task
    is launched as a background ``asyncio.Task`` and the call returns
    immediately with a SUCCESS string. Task references are kept in the
    module-level ``_swarm_task_registry`` keyed by a UUID so the agent can
    poll status from a later turn.

    Set ``await_results=True`` to block until every dispatched task
    completes; the aggregated per-task results (success or exception) are
    appended to the return string. This is useful when the agent has no
    other useful work to do until the swarm responds.
    """
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
            "output_key": output_key,
            "worker_persona": worker_persona
        })

    if not tasks:
        return "Error: No tasks provided to delegate_to_swarm."

    pretty_log("Swarm Dispatch", f"Delegating {len(tasks)} tasks to cluster", icon=Icons.BRAIN_PLAN)

    dispatched = 0
    skipped = 0
    invalid: list[str] = []
    dispatched_tasks: list[asyncio.Task] = []
    dispatched_keys: list[str] = []
    for task_def in tasks:
        t_instruction = task_def.get("instruction")
        t_input_data = task_def.get("input_data")
        t_output_key = task_def.get("output_key")
        t_worker_persona = task_def.get("worker_persona")
        t_target_model = task_def.get("target_model")

        if not t_instruction or not t_input_data or not t_output_key:
            pretty_log("Swarm Skip", f"Skipping invalid task definition: {task_def}", level="WARNING", icon=Icons.WARN)
            skipped += 1
            missing = [k for k, v in (
                ("instruction", t_instruction),
                ("input_data", t_input_data),
                ("output_key", t_output_key),
            ) if not v]
            invalid.append(f"missing={missing}")
            continue

        # Pre-validate that a routing node exists for this task BEFORE
        # spawning the background worker. Without this the tool returned
        # "SUCCESS: N tasks dispatched" even when the swarm had zero live
        # nodes — the model then planned against results that would never
        # arrive. Now we surface the failure synchronously.
        node = llm_client.get_swarm_node(t_target_model)
        if not node:
            skipped += 1
            invalid.append(f"no-node-for-model={t_target_model or 'any'}")
            # Best-effort: stash the failure in the scrapbook so the
            # model sees it there too on its next turn.
            try:
                scratchpad.set(
                    t_output_key,
                    f"SYSTEM ALERT: Swarm dispatch skipped for '{t_output_key}' — no matching edge node available.",
                )
            except Exception:
                pass
            continue

        task = asyncio.create_task(_swarm_worker(t_instruction, t_input_data, t_output_key, llm_client, model_name, scratchpad, worker_persona=t_worker_persona, target_model=t_target_model, preselected_node=node))
        _swarm_tasks.add(task)
        task.add_done_callback(_swarm_tasks.discard)

        # Register under a stable id so the agent can poll status from a
        # subsequent turn. The id is reported back in the return string.
        task_id = f"swarm-{uuid.uuid4().hex[:8]}"
        _register_task(task_id, task)

        # Best-effort: also stash on context.scratchpad if available so the
        # poll surface is the same as for swarm output.
        try:
            scratchpad.set(
                f"_swarm_task_id::{t_output_key}",
                task_id,
            )
        except Exception:
            pass

        dispatched_tasks.append(task)
        dispatched_keys.append(task_id)
        dispatched += 1

    if dispatched == 0:
        return (
            f"SYSTEM WARNING: 0 of {len(tasks)} task(s) dispatched to the Swarm "
            f"(reasons: {', '.join(invalid) if invalid else 'unknown'}). "
            f"You MUST process this synchronously — do not wait on the SCRAPBOOK."
        )

    # If the caller asked us to block, gather and report aggregated results.
    if await_results and dispatched_tasks:
        gathered = await asyncio.gather(*dispatched_tasks, return_exceptions=True)
        result_lines = []
        for tid, res in zip(dispatched_keys, gathered):
            if isinstance(res, BaseException):
                result_lines.append(f"  {tid}: ERROR {type(res).__name__}: {res}")
            else:
                result_lines.append(f"  {tid}: OK")
        body = "\n".join(result_lines) if result_lines else "(no results)"
        prefix = (
            f"SUCCESS: {dispatched}/{len(tasks)} task(s) completed (await_results=True)."
        )
        if skipped > 0:
            prefix = (
                f"PARTIAL: {dispatched}/{len(tasks)} task(s) completed; "
                f"{skipped} skipped ({', '.join(invalid)})."
            )
        return f"{prefix}\nTask IDs:\n{body}\nResults written to SCRAPBOOK at the requested output_key(s)."

    keys_str = ", ".join(dispatched_keys)
    if skipped > 0:
        return (
            f"PARTIAL: {dispatched}/{len(tasks)} task(s) dispatched; {skipped} skipped "
            f"({', '.join(invalid)}). Task IDs: {keys_str}. Check SCRAPBOOK for the "
            f"dispatched results; process the skipped items synchronously yourself."
        )
    return (
        f"SUCCESS: {dispatched} task(s) dispatched to the Swarm. Task IDs: {keys_str}. "
        f"The results will be silently written to your SCRAPBOOK when finished. "
        f"Do not wait—continue executing your next planned steps immediately."
    )
