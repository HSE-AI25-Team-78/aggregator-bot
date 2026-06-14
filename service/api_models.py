from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: Optional[float] = None

class HistoryItem(BaseModel):
    id: int
    text: Optional[str] = None
    label: Optional[str] = None
    timestamp: datetime
    work_time: float
    success: bool
    comment: Optional[str] = None
    version: str

class HistoryResponse(BaseModel):
    items: List[HistoryItem]
    count: int

class StatsResponse(BaseModel):
    request_stats: Dict[str, Any]
    input_stats: Dict[str, Any]


