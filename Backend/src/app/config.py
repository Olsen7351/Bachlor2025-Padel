import os
from functools import lru_cache
from pydantic_settings import BaseSettings


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
    secret_key: str = "dev-secret-key-change-in-production"
    api_title: str = "Padel Analyzer API"
    api_version: str = "0.1.0"

    # File Uploads
    upload_dir: str = "uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB

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

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
