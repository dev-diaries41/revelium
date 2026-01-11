from pydantic import BaseModel, Field
from typing import Optional
from openai.types import ResponsesModel
from smartscan import ClassificationResult

class LLMClientConfig(BaseModel):
    system_prompt: str
    model_name: str | ResponsesModel
    temperature: float = Field(default=0.1)
    max_output_tokens: int = Field(default=4000)
    stream: bool = Field(default=False)


class LLMClassificationResult(ClassificationResult):
    confidence: float
    
class Message(BaseModel):
    role: str
    content: Optional[str] = None