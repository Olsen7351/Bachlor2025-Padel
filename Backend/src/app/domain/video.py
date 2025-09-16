from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum

class VideoStatus(Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    ERROR = "error"

@dataclass
class Video:
    """Domain model for Video entity"""
    id: Optional[int]
    uploading_player_id: int
    file_name: str
    storage_path: str
    status: VideoStatus = VideoStatus.UPLOADED
    upload_timestamp: Optional[datetime] = None
    video_length: Optional[float] = None
    is_deleted: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None