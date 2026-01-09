from typing import Optional
from pydantic import  BaseModel, Field
from revelium.providers.types import TextEmbeddingModel
from openai.types import ResponsesModel
from revelium.constants import DEFAULT_CHROMADB_PATH, DEFAULT_OPENAI_MODEL, DEFAULT_PROMPTS_PATH, DEFAULT_SYSTEM_PROMPT

class ReveliumConfig(BaseModel):
    text_embedder: TextEmbeddingModel = Field(default="all-minilm-l6-v2")
    provider_model: ResponsesModel = Field(default=DEFAULT_OPENAI_MODEL)
    provider_api_key: Optional[str] = Field(default=None)
    chromadb_path: str = Field(default=DEFAULT_CHROMADB_PATH)
    prompt_store_path: str = Field(default=DEFAULT_PROMPTS_PATH)
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
    benchmarking: bool = Field(default=False)
