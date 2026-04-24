
import pytest
import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.agent import GhostAgent as Agent
from ghost_agent.core.planning import TaskStatus

@pytest.fixture
def mock_context():
    context = MagicMock()
    context.profile_memory = MagicMock()
    context.memory_system = MagicMock()
    context.skill_memory = MagicMock()
    context.llm_client = MagicMock()
    context.args = MagicMock()
    context.args.max_context = 8000
    context.args.temperature = 0.5
    context.args.use_planning = True
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = "No Data"
    return context

@pytest.mark.asyncio
async def test_planner_signals_completion_terminates_loop(mock_context):
    """
    Verifies that when the planner signals TaskStatus.DONE, the agent loop terminates immediately.
    """
    agent = Agent(mock_context)
    
    # Mock to_thread to avoid real IO
    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.return_value = "Mock Content"
        
        # 1. Setup LLM to return a plan that is DONE
        # We need a plan that sets the root task to DONE
        done_plan_json = {
            "thought": "The task is finished.",
            "tree_update": {
                "root_id": "root-123",
                "nodes": {
                    "root-123": {
                        "id": "root-123",
                        "description": "Main Task",
                        "status": "DONE", # <--- SIGNAL DONE
                        "children": []
                    }
                }
            },
            "next_action_id": "root-123"
        }
        
        # We want to ensure it doesn't loop infinitely.
        # If it didn't stop, it might try to call LLM again or execute tools.
        
        # Prepare JSON content
        json_str = str(done_plan_json).replace('None', 'null').replace('True', 'true').replace('False', 'false').replace("'", '"')
        mock_response = {
            "choices": [{
                "message": {
                    "content": f"```json\n{json_str}\n```"
                }
            }]
        }
        mock_context.llm_client.chat_completion.return_value = mock_response
        
        # Call handle_chat
        user_msg = "Do the thing"
        body = {"messages": [{"role": "user", "content": user_msg}]}
        background_tasks = MagicMock()
        
        # Execute
        await agent.handle_chat(body, background_tasks)
        
        # Verification
        # 1. The LLM should have been called distinct times.
        #    - Once for planning (which returns DONE)
        #    - Maybe once for the final response? 
        #    - Actually, if we force_stop, we break the loop. 
        #    - Then `handle_chat` continues to generate the final response if `final_ai_content` is empty?
        
        # Let's check if the loop ran more than once.
        # If force_stop works, it should break after turn 0 (or whichever turn the plan was generated).
        
        # We can simulate a sequence if needed, but here we just want to see if it stops.
        
        # If it stops, chat_completion call count for PLANNED tasks should be limited.
        # But wait, handle_chat has a loop `for turn in range(20)`.
        # Inside the loop:
        #   1. Plan (LLM call)
        #   2. Execute (if tools selected)
        #   3. If DONE -> force_stop = True -> break loop.
        
        # So we expect exactly 1 call to LLM for planning if it says DONE immediately.
        # And then maybe one call for the final response generation after the loop (if implemented that way).
        
        # Let's verify that `force_stop` logic was hit by ensuring we didn't loop again.
        
        # Verify LLM calls.
        assert mock_context.llm_client.chat_completion.call_count >= 1
        
        # If the loop continued, we would see more calls or tool executions.
        # Since we provided no tools in the DONE plan (just status update), 
        # if the loop didn't break, it might try to generate a tool call or speak?
        
        # Actually, `agent.py`:
        # ...
        # if task_tree.root_id and task_tree.nodes[task_tree.root_id].status == TaskStatus.DONE ... force_stop = True
        # ...
        # if force_stop: break
        
        # So it breaks the inner loop.
        # Then it proceeds to:
        # if not final_ai_content: generate final response.
        
        # So we expect:
        # 1. Planning call
        # 2. Final response generation call (since final_ai_content is likely empty)
        
        # Total calls = 2 or maybe 1 if it reuses something?
        # If logic was broken, it might loop 20 times!
        
        assert mock_context.llm_client.chat_completion.call_count <= 2, "Agent entered infinite loop or ran too many turns!"

