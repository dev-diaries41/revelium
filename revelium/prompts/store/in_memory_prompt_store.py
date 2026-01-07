from typing import List, Optional
from collections import OrderedDict
import asyncio
from revelium.prompts.types import Prompt
from revelium.prompts.store.prompt_store import PromptStore

class InMemoryPromptStore(PromptStore):
    """Async in-memory PromptStore using OrderedDict with asyncio.Lock for safe concurrent access."""

    def __init__(self):
        self._prompts: "OrderedDict[str, Prompt]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, limit: Optional[int] = None, offset: int = 0) -> List[Prompt]:
        async with self._lock:
            prompts_list = list(self._prompts.values())
            if limit is None:
                return prompts_list[offset:]
            return prompts_list[offset:offset + limit]

    async def get_by_ids(self, ids: List[str]) -> List[Prompt]:
        async with self._lock:
            return [self._prompts[p_id] for p_id in ids if p_id in self._prompts]

    async def add(self, items: List[Prompt]) -> None:
        async with self._lock:
            new_items = [item for item in items if item.prompt_id not in self._prompts]
            for item in new_items:
                self._prompts[item.prompt_id] = item

    async def delete(self, ids: List[str]) -> None:
        async with self._lock:
            for p_id in ids:
                self._prompts.pop(p_id, None)

    async def count(self) -> int:
        async with self._lock:
            return len(self._prompts)
