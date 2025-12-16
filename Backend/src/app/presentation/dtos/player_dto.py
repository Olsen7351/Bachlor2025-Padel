from pydantic import BaseModel, ConfigDict
from datetime import datetime

class PlayerResponse(BaseModel):
    """DTO for player response"""
    id: str
    name: str
    email: str
    role: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)