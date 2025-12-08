import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    # Database
    database_url: str = (
        "postgresql+asyncpg://padel_user:padel_password@localhost:5432/padel_dev"
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Environment
    environment: str = "development"

    # API
    api_title: str = "Padel Analyzer API"
    api_version: str = "0.1.0"

    # File Uploads - General
    upload_dir: str = "uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB (legacy - kept for compatibility)

    # Video Upload Settings
    video_upload_dir: str = "uploads/videos"
    video_max_file_size_mb: int = 2000  # 2 GB - allows 10-15 min videos at 1080p
    video_allowed_formats: list[str] = ["mp4", "avi", "mov", "mkv", "webm"]

    # Firebase Configuration
    firebase_project_id: Optional[str] = None
    firebase_private_key_id: Optional[str] = None
    firebase_private_key: Optional[str] = None
    firebase_client_email: Optional[str] = None
    firebase_client_id: Optional[str] = None
    firebase_web_api_key: Optional[str] = None

    # CORS Settings
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    model_config = ConfigDict(
        env_file = ".env",
        case_sensitive = False,
        extra = "ignore"
    )

    # Helper properties
    @property
    def video_max_file_size_bytes(self) -> int:
        """Get video max file size in bytes"""
        return self.video_max_file_size_mb * 1024 * 1024

    # Docker health check
    def is_database_available(self) -> bool:
        """Check if database is available"""
        try:
            import asyncpg
            import asyncio

            async def check():
                conn = await asyncpg.connect(self.database_url)
                await conn.close()
                return True

            return asyncio.run(check())
        except Exception:
            return False
    
    def validate_firebase_config(self) -> bool:
        """Check if Firebase is properly configured"""
        required_fields = [
            self.firebase_project_id,
            self.firebase_private_key_id, 
            self.firebase_private_key,
            self.firebase_client_email,
            self.firebase_client_id
        ]
        
        return all(field is not None and field.strip() != "" for field in required_fields)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Helper function to get Firebase configuration
def get_firebase_config() -> dict:
    """Get Firebase configuration as a dictionary"""
    settings = get_settings()
    
    if settings.validate_firebase_config():
        return {
            "type": "service_account",
            "project_id": settings.firebase_project_id,
            "private_key_id": settings.firebase_private_key_id,
            "private_key": settings.firebase_private_key.replace('\\n', '\n') if settings.firebase_private_key else None,
            "client_email": settings.firebase_client_email,
            "client_id": settings.firebase_client_id,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{settings.firebase_client_email.replace('@', '%40')}" if settings.firebase_client_email else None
        }
    
    raise ValueError("Firebase configuration not found. Please set Firebase environment variables or provide service account file path.")