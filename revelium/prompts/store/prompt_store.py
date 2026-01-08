from typing import List, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from revelium.prompts.types import Prompt


class PromptStore(ABC):
    """Abstract interface for storing and retrieving prompts."""

    @abstractmethod
    async def get(self, limit: Optional[int] = None, offset: int = 0) -> List[Prompt]:
        """Retrieve a list of prompts with optional pagination."""
        NotImplementedError("get method not implemented")

    @abstractmethod
    async def get_by_ids(self, ids: List[str]) -> List[Prompt]:
        """Retrieve prompts matching the given IDs."""
        NotImplementedError("get_by_ids method not implemented")

    @abstractmethod
    async def add(self, items: List[Prompt]) -> None:
        """Add multiple prompts to the store."""
        NotImplementedError("add method not implemented")

    @abstractmethod
    async def update(self, items: List[Prompt]) -> None:
        """Update multiple prompts."""
        NotImplementedError("update method not implemented")

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Delete prompts by their IDs."""
        NotImplementedError("delete method not implemented")

    @abstractmethod
    async def count(self, 
            cluster_id: Optional[str] = None,
            created_after: Optional[datetime] = None,
            created_before: Optional[datetime] = None,
            updated_after: Optional[datetime] = None,
            updated_before: Optional[datetime] = None,) -> int:        
          """Return the total number of prompts in the store."""
          NotImplementedError("count method not implemented")

