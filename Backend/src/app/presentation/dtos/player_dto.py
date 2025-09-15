from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional

class PlayerCreateRequest(BaseModel):
    """DTO for creating a player"""
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: Optional[str] = "player"

class PlayerResponse(BaseModel):
    """DTO for player response"""
    id: int
    name: str
    email: str
    role: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True