from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Prompt:
    prompt_id: str
    content: str
    cluster_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(datetime.timezone.utc))
    updated_at: Optional[datetime] = None