from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, BackgroundTasks, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
from pathlib import Path

# Business layer
from ...business.services.interfaces import IVideoService
from ...business.exceptions import (
    InvalidFileFormatException,
    FileTooLargeException,
    StorageException,
)
from ...domain.player import Player
from ..dtos.video_dto import (
    VideoUploadResponse,
    VideoErrorResponse,
    VideoStatusDto,
    PlayerVideosResponse,
    VideoSummaryDto
)

# DI container
from ...dependencies import get_video_service

# Database session - needed for commit/rollback in endpoint
from ...data.connection import get_db_session

from ...auth.dependencies import get_current_user


router = APIRouter(prefix="/videos", tags=["videos"])


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
    
    **Parameters:**
    - file: Video file to upload
    - court_number: Court number for calibration (required for ML analysis)
    
    **Video Processing:**
    - Videos larger than 1080p will be automatically downscaled
    - Videos with FPS > 30 will be converted to 30fps
    
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
    court_number: Annotated[int, Form(description="Court number for calibration")],
    current_user: Player = Depends(get_current_user),
    video_service: IVideoService = Depends(get_video_service),
    session: AsyncSession = Depends(get_db_session)
) -> VideoUploadResponse:
    """
    UC-01: Upload video file
    
    Handles:
    - File validation (format and size)
    - Video conversion (if needed)
    - File storage
    - Database record creation

    Triggers:
    - Background ML analysis with court_number
    """
    
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Validate court number
    if court_number < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Court number must be a positive integer"
        )
    
    # Get file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    try:
        # Upload video through service (handles conversion)
        video = await video_service.upload_video(
            file=file.file,
            filename=file.filename,
            content_type=file.content_type,
            file_size=file_size,
            player_id=current_user.id
        )
        
        await session.commit()

        # Get video file path for analysis
        video_path = video_service.get_video_path(video)

        # Trigger analysis in background with court_number
        background_tasks.add_task(
            process_video_analysis,
            video_id=video.id,
            player_id=current_user.id,
            court_number=court_number,
            video_path=str(video_path)
        )

        return VideoUploadResponse(
            id=video.id,
            file_name=video.file_name,
            status=VideoStatusDto(video.status.value),
            upload_timestamp=video.upload_timestamp,
            video_length=video.video_length,
            message="Video uploaded successfully. Analysis will start shortly."
        )
        
    except InvalidFileFormatException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(e),
                "supported_formats": video_service.get_allowed_formats(),
                "max_size_mb": video_service.get_max_file_size_mb()
            }
        )
    
    except FileTooLargeException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": str(e),
                "max_size_mb": video_service.get_max_file_size_mb(),
                "supported_formats": video_service.get_allowed_formats()
            }
        )
    
    except StorageException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to store video file",
                "details": str(e)
            }
        )
    
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "An unexpected error occurred",
                "details": str(e)
            }
        )
    

@router.get(
        "/myVideos",
        response_model=PlayerVideosResponse,
        status_code=status.HTTP_200_OK,
        summary="Get player's videos"
)
async def get_player_videos(
    current_user: Player = Depends(get_current_user),
    video_service: IVideoService = Depends(get_video_service)
) -> PlayerVideosResponse:
    """
    Get all videos for the authenticated player.
    """
    videos = await video_service.get_player_videos(
        player_id=current_user.id
    )

    video_summaries = [
        VideoSummaryDto(
            id=video.id,
            file_name=video.file_name,
            status=VideoStatusDto(video.status.value),
            upload_timestamp=video.upload_timestamp,
            video_length=video.video_length
        )
        for video in videos
    ]

    return PlayerVideosResponse(
        videos=video_summaries,
        total_count=len(video_summaries)
    )


async def process_video_analysis(
    video_id: int, 
    player_id: str, 
    court_number: int,
    video_path: str
):
    """
    Background task to run ML analysis.
    
    Note: Background tasks manage their own database session lifecycle,
    so we use the factory function from the DI container that accepts
    a session directly.
    """
    # Import here to avoid circular imports in background task
    from app.data.connection import get_db_session
    from app.dependencies import create_analysis_service
    
    async for session in get_db_session():
        try:
            # Use the DI container's factory function
            analysis_service = create_analysis_service(session)
            
            # Step 1: Create analysis entity chain
            analysis = await analysis_service.create_analysis_for_video(
                video_id=video_id,
                player_id=player_id
            )

            # Save match_id so the object can be detached
            match_id = analysis.match_id

            # Commit analysis creation
            await session.commit()
            
            # Step 2: Run ML analysis
            ml_result = await analysis_service.run_ml_analysis(
                video_id=video_id,
                video_path=Path(video_path),
                court_number=court_number
            )
            
            # Step 3: Store results
            await analysis_service.store_analysis_results(
                match_id=match_id,
                ml_result=ml_result
            )
            
            # Step 4: Mark as complete
            await analysis_service.complete_analysis(video_id, success=True)
            
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