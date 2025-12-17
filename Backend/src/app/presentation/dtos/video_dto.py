from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, List
from enum import Enum


class VideoStatusDto(str, Enum):
    """Video processing status"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    ERROR = "error"


class VideoUploadResponse(BaseModel):
    """Response after successful video upload"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    file_name: str
    status: VideoStatusDto
    upload_timestamp: datetime
    video_length: Optional[float] = None
    message: str = "Video uploaded successfully"


class VideoSummaryDto(BaseModel):
    """Summary of a single video for list display"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_name: str
    status: VideoStatusDto
    upload_timestamp: datetime
    video_length: Optional[float] = None


class PlayerVideosResponse(BaseModel):
    """Response containing list of player's analyzed videos"""
    model_config = ConfigDict(from_attributes=True)

    videos: List[VideoSummaryDto]
    total_count: int


class VideoErrorResponse(BaseModel):
    """Error response for video operations"""
    error: str
    details: Optional[str] = None
    supported_formats: Optional[list[str]] = None
    max_size_mb: Optional[int] = None
