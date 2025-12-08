from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Analysis:
    """Domain model for Analysis entity"""
    id: Optional[int]
    player_id: str  # Firebase string ID
    video_id: int
    match_id: int
    analysis_timestamp: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None