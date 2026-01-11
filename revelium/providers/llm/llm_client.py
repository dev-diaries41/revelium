from abc import abstractmethod, ABC
from typing import List, Optional, Dict, TypeVar, Type
from pydantic import BaseModel
from revelium.schemas.llm import Message
import numpy as np

JsonOutput = TypeVar("JsonOutput", bound=BaseModel)

class LLMClient(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, history: Optional[List[Message]] = None) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def generate_json(self, prompt: str, format: Type[JsonOutput], history: Optional[List[Message]] = None) -> JsonOutput: 
        raise NotImplementedError
    
    @abstractmethod
    def generate_text_from_image(self, prompt: str, images: List[Dict[str, str]], history: Optional[List[Message]] = None) -> np.ndarray:
        raise NotImplementedError
