from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from .player_dto import PlayerResponse

class RegisterRequest(BaseModel):
    """DTO for user registration after Firebase auth"""
    name: str = Field(
        ...,
        min_length=1, 
        max_length=100, 
        description="Player name"
        )

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "name": "John Doe"
            }
        }
    )
        

class LoginResponse(BaseModel):
    """DTO for successful login response"""
    message: str
    user: PlayerResponse
    
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "message": "Login successful",
                "user": {
                    "id": "firebase-uid-123",
                    "name": "John Doe",
                    "email": "john@example.com",
                    "role": "player",
                    "created_at": "2024-09-24T12:00:00",
                    "updated_at": "2024-09-24T12:00:00"
                }
            }
        }
    )

