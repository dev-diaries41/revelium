from datetime import datetime, timezone
from pydantic import BaseModel, Field

class PromptMetadata(BaseModel):
    UNCLUSTERED: str = "Unclustered"
    cluster_id: str = UNCLUSTERED
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Prompt(BaseModel):
    prompt_id: str
    content: str
    metadata: PromptMetadata = Field(default_factory=PromptMetadata)
