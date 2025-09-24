from pydantic import BaseModel, EmailStr
from typing import Optional
from .player_dto import PlayerResponse

class RegisterRequest(BaseModel):
    """DTO for user registration after Firebase auth"""
    name: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe"
            }
        }

class LoginResponse(BaseModel):
    """DTO for successful login response"""
    message: str
    user: PlayerResponse
    
    class Config:
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

class AuthUserInfo(BaseModel):
    """DTO for current user info"""
    uid: str
    email: str
    email_verified: bool
    name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "uid": "firebase-uid-123",
                "email": "john@example.com", 
                "email_verified": True,
                "name": "John Doe"
            }
        }
