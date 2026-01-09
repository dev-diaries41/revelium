from pydantic import BaseModel
from typing import Literal

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
