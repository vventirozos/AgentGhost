from typing import Iterator, Dict, Any

from qwen_agent.agents import ReActChat
from src.ghost_agent.tools.qwen_bridge import set_context
from src.ghost_agent.core.prompts import SYSTEM_PROMPT

class GhostQwenAgent(ReActChat):
    def __init__(self, context: Any, llm_cfg: Dict[str, Any]):
        set_context(context)
        
        super().__init__(
            function_list=['file_system', 'execute', 'knowledge_base'],
            llm=llm_cfg,
            system_message=SYSTEM_PROMPT
        )
        
    def _run(self, messages: list[Dict[str, Any]], lang: str = 'en', **kwargs: Any) -> Iterator[list[Dict[str, Any]]]:
        yield from super()._run(messages=messages, lang=lang, **kwargs)
