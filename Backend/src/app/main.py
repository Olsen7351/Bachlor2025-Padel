from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.data.connection import create_tables
from app.config import get_settings
from app.presentation.controllers.player_controller import router as player_router
from datetime import datetime

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

app = FastAPI(
    title=settings.api_title, 
    version=settings.api_version, 
    lifespan=lifespan
)

# Include API routes
app.include_router(player_router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "Padel Analyzer API - 3-Tier Architecture",
        "version": settings.api_version,
        "environment": settings.environment,
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}