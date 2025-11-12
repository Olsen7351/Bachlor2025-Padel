from pydantic import BaseModel, ConfigDict
from .player_dto import PlayerResponse

        
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

