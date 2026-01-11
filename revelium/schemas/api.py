from pydantic import BaseModel
from typing import List, Optional, Literal

from smartscan import ClusterMetadata

from revelium.prompts.types import Prompt

# Websocksets / SSE
class ActiveMessage(BaseModel):
    type: Literal["active"] = "active"
    
class ProgressMessage(BaseModel):
    type: Literal["progress"] = "progress"
    progress: float

class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    error: str
    item: str

class FailMessage(BaseModel):
    type: Literal["fail"] = "fail"
    error: str

class CompleteMessage(BaseModel):
    type: Literal["complete"] = "complete"
    total_processed: int
    time_elapsed: float

# Long running jobs
FinishedStatus = Literal['completed', 'failed']

Status = Literal[
    FinishedStatus,
    "active",
    "delayed",
    "prioritized",
    "waiting",
    "waiting-children",
]

class JobReceipt(BaseModel):
    jobId: str
    status: Status
    queue: int
    createdAt: str # may change to float / datetime
    delay: float
    jobName: Optional[str] = None

# HTTP
class PromptsPayload(BaseModel):
    prompts: List[Prompt]

class AddPromptsRequest(PromptsPayload):
    pass

class AddPromptsResponse(JobReceipt):
    pass
 

class GetPromptsRequest(BaseModel):
    prompt_ids: List[str]

class GetPromptsResponse(PromptsPayload):
    pass

class GetCountResponse(BaseModel):
    count: int

class GetLabelsResponse(BaseModel):
    labels: List[str]

class GetClusterMetadataResponse(BaseModel):
    metadata: Optional[ClusterMetadata]
    

    
    

