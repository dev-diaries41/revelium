from typing import List, Optional
from abc import ABC, abstractmethod
from revelium.prompts.types import Prompt


class PromptStore(ABC):
    """Abstract interface for storing and retrieving prompts."""

    @abstractmethod
    def get(self, limit: Optional[int] = None, offset: int = 0) -> List[Prompt]:
        """Retrieve a list of prompts with optional pagination."""
        pass

    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> List[Prompt]:
        """Retrieve prompts matching the given IDs."""
        pass

    @abstractmethod
    def add(self, items: List[Prompt]) -> None:
        """Add multiple prompts to the store."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete prompts by their IDs."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of prompts in the store."""
        pass
