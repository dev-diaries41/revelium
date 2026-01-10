from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class PromptMetadata:
    UNCLUSTERED ="Unclustered"
    cluster_id: str = UNCLUSTERED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Prompt:
    prompt_id: str
    content: str
    metadata: PromptMetadata = field(default_factory=PromptMetadata)
   
