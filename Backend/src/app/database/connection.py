from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.models.database import Base
from app.config import get_settings

settings = get_settings()

# Create async engine based on environment
if settings.environment == "production":
    # Production settings - add connection pooling, etc.
    engine = create_async_engine(
        settings.database_url,
        echo=False,  # Don't log SQL in production
        pool_size=20,
        max_overflow=0,
    )
else:
    # Development settings
    engine = create_async_engine(
        settings.database_url,
        echo=True,  # Log SQL queries for debugging
    )

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db_session() -> AsyncSession:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_tables():
    """Create all tables - use Alembic migrations in production"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables():
    """Drop all tables - for development only"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
