"""Video controller - Presentation layer (FastAPI endpoints)"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from app.presentation.dtos.video_dto import (
    VideoUploadResponse,
    VideoErrorResponse,
    VideoStatusDto
)
from ...business.services.video_service import VideoService
from ...business.services.file_storage import FileStorageService
from ...data.repositories.video_repository import VideoRepository
from ...data.connection import get_db_session
from ...auth.dependencies import get_current_user
from ...domain.player import Player
from ...business.exceptions import (
    InvalidFileFormatException,
    FileTooLargeException,
    StorageException,
)


router = APIRouter(prefix="/videos", tags=["videos"])


def get_video_service(session: AsyncSession = Depends(get_db_session)) -> VideoService:
    """
    Dependency injection for VideoService
    Follows Dependency Inversion Principle
    """
    video_repository = VideoRepository(session)
    file_storage_service = FileStorageService()
    return VideoService(video_repository, file_storage_service)


@router.post(
    "/upload",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": VideoErrorResponse, "description": "Invalid file format or size"},
        500: {"model": VideoErrorResponse, "description": "Storage or server error"}
    },
    summary="Upload a padel match video",
    description="""
    Upload a video file for analysis. Implements UC-01.
    
    **Success Scenarios:**
    - S1: Video uploaded successfully and analysis will start
    - S2: Video queued while another video is being analyzed
    
    **Failure Scenarios:**
    - F1: File format not supported
    - F2: File size exceeds limit
    - F3: Network/storage error during upload
    
    **Supported formats:** MP4, AVI, MOV, MKV, WEBM  
    **Maximum size:** 2000 MB
    """
)
async def upload_video(
    file: Annotated[UploadFile, File(description="Video file to upload")],
    background_tasks: BackgroundTasks,
    current_user: Player = Depends(get_current_user),
    video_service: VideoService = Depends(get_video_service),
    session: AsyncSession = Depends(get_db_session)
) -> VideoUploadResponse:
    """
    UC-01: Upload video file
    
    Handles:
    - File validation (format and size)
    - File storage
    - Database record creation
    - Success and failure scenarios from UC-01

    Triggers:
    - New analysis
    """
    
    # Validate that a file was uploaded
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Get file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()  # Get position (size)
    file.file.seek(0)  # Reset to beginning
    
    try:
        # Upload video through service (Business Logic Layer)
        video = await video_service.upload_video(
            file=file.file,
            filename=file.filename,
            content_type=file.content_type,
            file_size=file_size,
            player_id=current_user.id
        )
        
        await session.commit()

        # Trigger analysis in background
        background_tasks.add_task(
            process_video_analysis,
            video_id=video.id,
            player_id=current_user.id
        )

        # Map domain entity to DTO
        return VideoUploadResponse(
            id=video.id,
            file_name=video.file_name,
            status=VideoStatusDto(video.status.value),
            upload_timestamp=video.upload_timestamp,
            video_length=video.video_length,
            message="Video uploaded successfully. Analysis will start shortly."
        )
        
    except InvalidFileFormatException as e:
        # F1: File format not supported
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(e),
                "supported_formats": video_service.get_allowed_formats(),
                "max_size_mb": video_service.get_max_file_size_mb()
            }
        )
    
    except FileTooLargeException as e:
        # F2: File too large
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(e),
                "max_size_mb": video_service.get_max_file_size_mb(),
                "supported_formats": video_service.get_allowed_formats()
            }
        )
    
    except StorageException as e:
        # F3: Storage error (network/disk issues)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to store video file",
                "details": str(e)
            }
        )
    
    except Exception as e:
        # Unexpected errors
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "An unexpected error occurred",
                "details": str(e)
            }
        )


@router.get(
    "/config/upload-info",
    summary="Get upload configuration",
    description="Get information about allowed file formats and size limits"
)
async def get_upload_config(
    video_service: VideoService = Depends(get_video_service)
) -> dict:
    """Get upload configuration information"""
    
    return {
        "max_file_size_mb": video_service.get_max_file_size_mb(),
        "allowed_formats": video_service.get_allowed_formats(),
        "description": "Upload configuration for video files"
    }

async def process_video_analysis(video_id: int, player_id: str):
    """Background task to create analysis and run ML model"""
    from app.data.connection import get_db_session
    from app.data.repositories.analysis_repository import AnalysisRepository
    from app.data.repositories.match_repository import (
        MatchRepository, MatchPlayerRepository
    )
    from app.data.repositories.summary_metrics_repository import SummaryMetricsRepository
    from app.data.repositories.video_repository import VideoRepository
    from app.business.services.analysis_service import AnalysisService
    import asyncio
    
    # Get database session
    async for session in get_db_session():
        try:
            # Create repositories
            analysis_repo = AnalysisRepository(session)
            match_repo = MatchRepository(session)
            match_player_repo = MatchPlayerRepository(session)
            metrics_repo = SummaryMetricsRepository(session)
            video_repo = VideoRepository(session)
            
            # Create analysis service
            analysis_service = AnalysisService(
                analysis_repo,
                match_repo,
                match_player_repo,
                metrics_repo,
                video_repo
            )
            
            # Step 1: Create analysis entity chain
            analysis = await analysis_service.create_analysis_for_video(
                video_id=video_id,
                player_id=player_id
            )
            
            # Step 2: Run AI analysis (mock for now)
            ai_results = await run_mock_ai_analysis(video_id)
            
            # Step 3: Store results
            await analysis_service.store_analysis_results(
                match_id=analysis.match_id,
                ai_results=ai_results
            )
            
            # Step 4: Mark as complete
            await analysis_service.complete_analysis(video_id, success=True)
            
            # Commit transaction
            await session.commit()
            
        except Exception as e:
            print(f"Analysis failed for video {video_id}: {str(e)}")
            try:
                await analysis_service.complete_analysis(
                    video_id, 
                    success=False
                )
                await session.commit()
            except:
                await session.rollback()


async def run_mock_ai_analysis(video_id: int) -> dict:
    """
    Mock AI analysis - REPLACE THIS with real ML model later
    Returns fake hit counts for testing
    """
    import asyncio
    import random
    
    # Simulate AI processing time (2 seconds)
    await asyncio.sleep(2)
    
    # Return mock results
    return {
        "player_1": {"hits": random.randint(200, 250), "rallies": 45},
        "player_2": {"hits": random.randint(200, 250), "rallies": 45},
        "player_3": {"hits": random.randint(180, 230), "rallies": 45},
        "player_4": {"hits": random.randint(180, 230), "rallies": 45}
    }