from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.data.connection import create_tables
from app.config import get_settings
from app.presentation.controllers.player_controller import router as player_router
from app.presentation.controllers.auth_controller import router as auth_router
from app.business.exceptions import AuthenticationException, PlayerNotFoundException, ValidationException
from datetime import datetime

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - create database tables
    print("🚀 Starting up...")
    try:
        await create_tables()
        print("✅ Database tables created/verified")
        
        # Test Firebase configuration (optional)
        try:
            from app.auth.firebase_service import FirebaseService
            FirebaseService()  # This will initialize Firebase and print status
        except Exception as e:
            print(f"⚠️  Firebase initialization failed: {e}")
            print("   Make sure Firebase environment variables are set correctly")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
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

@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "validation_error"}
    )

# Include API routes
app.include_router(auth_router, prefix="/api")      # Auth routes
app.include_router(player_router, prefix="/api")    # Protected player routes

@app.get("/")
async def root():
    return {
        "message": "Padel Analyzer API",
        "version": settings.api_version,
        "environment": settings.environment,
        "auth_required": True,
        "firebase_configured": settings.validate_firebase_config()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow(),
        "database": settings.is_database_available(),
        "firebase": settings.validate_firebase_config()
    }