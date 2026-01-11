from typing import Optional, ClassVar
from pydantic import  BaseModel, Field
from openai.types import ResponsesModel
from revelium.providers.types import TextEmbeddingModel
from revelium.constants import BASE_DIR

class ReveliumConfig(BaseModel):
    DEFAULT_CHROMADB_PATH: ClassVar[str] = BASE_DIR / "revelium_chromadb"
    DEFAULT_OPENAI_MODEL: ClassVar[str] = "gpt-5-mini"
    DEFAULT_TEXT_EMBEDDER: ClassVar[str] = "all-minilm-l6-v2"
    text_embedder: TextEmbeddingModel = Field(default=DEFAULT_TEXT_EMBEDDER)
    provider_model: ResponsesModel = Field(default=DEFAULT_OPENAI_MODEL)
    provider_api_key: Optional[str] = Field(default=None)
    chromadb_path: str = Field(default=DEFAULT_CHROMADB_PATH)
    benchmarking: bool = Field(default=False)
