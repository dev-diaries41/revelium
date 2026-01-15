

from enum import Enum
from typing import Dict, Optional, Union

class ErrorCode(Enum):
    """Enum for error codes used by client."""

    PROMPT_NOT_FOUND = "PROMPT_NOT_FOUND"
    INVALID_MODEL_PATH = "INVALID_MODEL_PATH"
    MISSING_API_KEY = "MISSING_API_KEY"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    MISSING_LLM_CLIENT = "MISSING_LLM_CLIENT"

class ReveliumError(Exception):
    """Base class for all SmartScan related errors."""

    def __init__(self, message: str, code: Optional[ErrorCode] = None, details: Optional[Union[Dict, str, object]] = None):
        if details is None:
            details = {}
        self.message = message
        self.code = code
        self.details = details
        super().__init__(message)