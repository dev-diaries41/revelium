from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Prompt:
    prompt_id: str
    content: str
    created_at: datetime = field(default_factory=datetime.now(datetime.timezone.utc))
    updated_at: Optional[datetime] = None
