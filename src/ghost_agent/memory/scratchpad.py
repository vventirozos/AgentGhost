from typing import Dict, Any, Optional
from collections import OrderedDict

class Scratchpad:
    def __init__(self, max_entries: int = 50):
        self._data = OrderedDict()
        self.max_entries = max_entries

    def set(self, key: str, value: Any):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        
        # Evict oldest if over capacity
        if len(self._data) > self.max_entries:
            self._data.popitem(last=False)
            
        return f"Stored: {key} = {value}"

    def get(self, key: str) -> Optional[Any]:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def list_all(self) -> str:
        if not self._data:
            return "Scratchpad is empty."
        return "\n".join([f"{k}: {v}" for k, v in self._data.items()])

    def clear(self):
        self._data.clear()
        return "Scratchpad cleared."