from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

from .data.connection import create_tables
from .config import get_settings
from .presentation.controllers.auth_controller import router as auth_router
from .presentation.controllers.video_controller import router as video_router
from .presentation.controllers.match_controller import router as match_router
from .presentation.controllers.heatmap_controller import router as heatmap_router
from .presentation.controllers.rally_controller import router as rally_router
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
    AnalysisNotCompleteException,
    HeatmapNotFoundException,
    InsufficientPositionDataException,
    RallyDataUnavailableException
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events"""
    # Startup
    print("Starting up...")
    try:
        await create_tables()
        print("Database tables created/verified")
        
        # Ensure upload directories exist
        import os
        os.makedirs(settings.video_upload_dir, exist_ok=True)
        print(f"Upload directory created/verified: {settings.video_upload_dir}")
        
        # Test Firebase configuration
        try:
            from app.auth.firebase_service import FirebaseService
            FirebaseService()
        except Exception as e:
            print(f"Firebase initialization failed: {e}")
            print("Make sure Firebase environment variables are set correctly")
        
    except Exception as e:
        print(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title=settings.api_title, 
    version=settings.api_version, 
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global Exception Handlers
# ============================================================================

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
    """UC-04 F1, UC-02 F1: Handle when analysis data is not available"""
    return JSONResponse(
        status_code=503,
        content={
            "detail": str(exc),
            "type": "data_unavailable",
            "reason": "Analysis may have failed or is incomplete"
        }
    )


@app.exception_handler(AnalysisNotCompleteException)
async def analysis_not_complete_handler(request: Request, exc: AnalysisNotCompleteException):
    return JSONResponse(
        status_code=409,
        content={
            "detail": str(exc),
            "type": "analysis_not_complete",
            "message": "Analysis is still in progress or has not started"
        }
    )


@app.exception_handler(HeatmapNotFoundException)
async def heatmap_not_found_handler(request: Request, exc: HeatmapNotFoundException):
    """UC-02: Handle when heatmap data is not found"""
    return JSONResponse(
        status_code=404,
        content={
            "detail": str(exc),
            "type": "heatmap_not_found"
        }
    )


@app.exception_handler(InsufficientPositionDataException)
async def insufficient_position_data_handler(request: Request, exc: InsufficientPositionDataException):
    """UC-02 F2: Handle when AI couldn't track player sufficiently"""
    return JSONResponse(
        status_code=503,
        content={
            "detail": str(exc),
            "type": "insufficient_position_data",
            "reason": "The AI could not track this player sufficiently to generate a heatmap"
        }
    )

@app.exception_handler(RallyDataUnavailableException)
async def rally_data_unavailable_handler(request: Request, exc: RallyDataUnavailableException):
    """UC-08 F1: Handle when AI couldn't distinguish individual rallies"""
    return JSONResponse(
        status_code=503,
        content={
            "detail": str(exc),
            "type": "rally_data_unavailable",
            "reason": "The AI could not distinguish individual rallies from each other"
        }
    )


# ============================================================================
# Include API Routers
# ============================================================================

app.include_router(auth_router, prefix="/api")
app.include_router(video_router, prefix="/api")
app.include_router(match_router, prefix="/api")
app.include_router(heatmap_router, prefix="/api")
app.include_router(rally_router, prefix="/api")


# ============================================================================
# Default Endpoints
# ============================================================================

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
            "matches": "/api/matches",
            "heatmaps": "/api/heatmaps",
            "rallies": "/api/rallies"
        }
    }