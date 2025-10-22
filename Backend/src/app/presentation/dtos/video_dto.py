from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
from typing import Optional
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


class VideoErrorResponse(BaseModel):
    """Error response for video operations"""
    error: str
    details: Optional[str] = None
    supported_formats: Optional[list[str]] = None
    max_size_mb: Optional[int] = None