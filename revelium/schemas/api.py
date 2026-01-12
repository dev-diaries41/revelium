from pydantic import BaseModel
from typing import List, Optional, Literal, ClassVar

from smartscan import Cluster, ClusterMetadata

from revelium.prompts.types import Prompt, PromptsOverviewInfo

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

class GetPromptsOverviewResponse(PromptsOverviewInfo):
    pass

class GetCountResponse(BaseModel):
    count: int

class GetLabelsResponse(BaseModel):
    labels: List[str]

class GetClusterRequestParams(BaseModel):
    cluster_id: Optional[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = None


class ClusterNoEmbeddings(BaseModel):
    UNLABELLED:ClassVar[str] = "unlabelled"
    prototype_id: str
    metadata: ClusterMetadata
    label: str
    
class GetClustersResponse(BaseModel):
    clusters: List[ClusterNoEmbeddings]

