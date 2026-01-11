from typing import Literal, TypeAlias, TypedDict, Optional

LocalTextEmbeddingModel: TypeAlias = Literal["all-minilm-l6-v2",]

TextEmbeddingModel: TypeAlias = Literal[
    LocalTextEmbeddingModel,
    "text-embedding-3-small",
    "text-embedding-3-large",
]

class ModelInfo(TypedDict):
    url: str
    path: str
    sha256: Optional[str] = None