from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.database.connection import create_tables
from app.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - create database tables
    print("Starting up...")
    try:
        await create_tables()
        print("Database tables created/verified")
    except Exception as e:
        print(f"Database setup failed: {e}")
        raise

    yield

    # Shutdown
    print("Shutting down...")


app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)

# Include API routes
app.include_router(router, prefix="/api")
