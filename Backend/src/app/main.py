from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .data.connection import create_tables
from .config import get_settings
from .presentation.controllers.player_controller import router as player_router
from .presentation.controllers.auth_controller import router as auth_router
from .presentation.controllers.video_controller import router as video_router
from .presentation.controllers.match_controller import router as match_router
from .business.exceptions import (
    AuthenticationException, 
    PlayerNotFoundException, 
    ValidationException,
    VideoNotFoundException,
    InvalidFileFormatException,
    FileTooLargeException,
    StorageException,
    MatchNotFoundException,
    PlayerInMatchNotFoundException,
    DataUnavailableException,
    InvalidSetNumberException,
    AnalysisNotCompleteException
)
from datetime import datetime

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - create database tables
    print("🚀 Starting up...")
    try:
        await create_tables()
        print("✅ Database tables created/verified")
        
        # Ensure upload directories exist
        import os
        os.makedirs(settings.video_upload_dir, exist_ok=True)
        print(f"✅ Upload directory created/verified: {settings.video_upload_dir}")
        
        # Test Firebase configuration (optional)
        try:
            from app.auth.firebase_service import FirebaseService
            FirebaseService()  # This will initialize Firebase and print status
        except Exception as e:
            print(f"⚠️  Firebase initialization failed: {e}")
            print("   Make sure Firebase environment variables are set correctly")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

    yield

    # Shutdown
    print("👋 Shutting down...")


app = FastAPI(
    title=settings.api_title, 
    version=settings.api_version, 
    lifespan=lifespan
)

# CORS middleware using config settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handlers
@app.exception_handler(AuthenticationException)
async def auth_exception_handler(request: Request, exc: AuthenticationException):
    return JSONResponse(
        status_code=401,
        content={"detail": str(exc), "type": "authentication_error"}
    )


@app.exception_handler(PlayerNotFoundException)
async def player_not_found_handler(request: Request, exc: PlayerNotFoundException):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "type": "not_found_error"}
    )


@app.exception_handler(VideoNotFoundException)
async def video_not_found_handler(request: Request, exc: VideoNotFoundException):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "type": "not_found_error"}
    )


@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "validation_error"}
    )


@app.exception_handler(InvalidFileFormatException)
async def invalid_file_format_handler(request: Request, exc: InvalidFileFormatException):
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc), 
            "type": "invalid_file_format",
            "allowed_formats": settings.video_allowed_formats,
            "max_size_mb": settings.video_max_file_size_mb
        }
    )


@app.exception_handler(FileTooLargeException)
async def file_too_large_handler(request: Request, exc: FileTooLargeException):
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
            "type": "file_too_large",
            "max_size_mb": settings.video_max_file_size_mb
        }
    )


@app.exception_handler(StorageException)
async def storage_exception_handler(request: Request, exc: StorageException):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Failed to store file",
            "type": "storage_error",
            "message": str(exc)
        }
    )


@app.exception_handler(MatchNotFoundException)
async def match_not_found_handler(request: Request, exc: MatchNotFoundException):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "type": "match_not_found"}
    )


@app.exception_handler(PlayerInMatchNotFoundException)
async def player_in_match_not_found_handler(request: Request, exc: PlayerInMatchNotFoundException):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "type": "player_in_match_not_found"}
    )


@app.exception_handler(DataUnavailableException)
async def data_unavailable_handler(request: Request, exc: DataUnavailableException):
    """UC-04 F1: Handle when hit data is not available"""
    return JSONResponse(
        status_code=503,  # Service Unavailable
        content={
            "detail": str(exc),
            "type": "data_unavailable",
            "reason": "Analysis may have failed or is incomplete"
        }
    )


@app.exception_handler(InvalidSetNumberException)
async def invalid_set_number_handler(request: Request, exc: InvalidSetNumberException):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "invalid_set_number"}
    )


@app.exception_handler(AnalysisNotCompleteException)
async def analysis_not_complete_handler(request: Request, exc: AnalysisNotCompleteException):
    return JSONResponse(
        status_code=409,  # Conflict
        content={
            "detail": str(exc),
            "type": "analysis_not_complete",
            "message": "Analysis is still in progress or has not started"
        }
    )


# Include API routes
app.include_router(auth_router, prefix="/api")      # Auth routes
app.include_router(player_router, prefix="/api")    # Protected player routes
app.include_router(video_router, prefix="/api")     # Video upload routes
app.include_router(match_router, prefix="/api")     # Match routes


@app.get("/")
async def root():
    return {
        "message": "Padel Analyzer API",
        "version": settings.api_version,
        "environment": settings.environment,
        "auth_required": True,
        "firebase_configured": settings.validate_firebase_config(),
        "endpoints": {
            "auth": "/api/auth",
            "players": "/api/players",
            "videos": "/api/videos",
            "matches": "/api/matches"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow(),
        "database": settings.is_database_available(),
        "firebase": settings.validate_firebase_config(),
        "upload_directory": settings.video_upload_dir
    }


@app.get("/api/config/upload")
async def get_upload_config():
    """Public endpoint to get upload configuration"""
    return {
        "max_file_size_mb": settings.video_max_file_size_mb,
        "max_file_size_bytes": settings.video_max_file_size_bytes,
        "allowed_formats": settings.video_allowed_formats,
        "upload_directory": settings.video_upload_dir
    }