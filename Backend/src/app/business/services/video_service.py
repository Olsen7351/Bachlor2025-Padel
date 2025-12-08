from typing import BinaryIO, Optional
from pathlib import Path
from datetime import datetime
import ffmpeg

from ...domain.video import Video, VideoStatus
from ...business.services.interfaces import IVideoService, IFileStorageService
from ...data.repositories.interfaces import IVideoRepository
from ...business.exceptions import (
    InvalidFileFormatException,
    FileTooLargeException,
    VideoNotFoundException,
    StorageException
)
from ...config import get_settings


class VideoService(IVideoService):
    """
    Video service implementation
    Handles business logic for video operations
    """
    
    def __init__(
        self, 
        video_repository: IVideoRepository,
        file_storage_service: IFileStorageService
    ):
        """
        Initialize service with dependencies (Dependency Inversion Principle)
        
        Args:
            video_repository: Repository interface for video data access
            file_storage_service: Storage service interface for file operations
        """
        self._video_repository = video_repository
        self._file_storage = file_storage_service
        self._settings = get_settings()
    
    @property
    def ALLOWED_VIDEO_FORMATS(self) -> list[str]:
        """Get allowed video formats from configuration"""
        return self._settings.video_allowed_formats
    
    @property
    def MAX_FILE_SIZE_MB(self) -> int:
        """Get max file size in MB from configuration"""
        return self._settings.video_max_file_size_mb
    
    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        """Get max file size in bytes from configuration"""
        return self._settings.video_max_file_size_bytes
    
    async def upload_video(
        self, 
        file: BinaryIO, 
        filename: str, 
        content_type: str,
        file_size: int,
        player_id: str
    ) -> Video:
        """
        Upload and process a video file
        Implements UC-01 Success Scenario S1 and Failure Scenarios F1, F2
        
        Business Rules:
        1. Validate file format (F1)
        2. Validate file size (F2)
        3. Store file securely
        4. Create database record
        5. Set initial status to UPLOADED
        
        Returns:
            Video domain entity with UPLOADED status
            
        Raises:
            InvalidFileFormatException: If file format is not supported (F1)
            FileTooLargeException: If file exceeds size limit (F2)
            StorageException: If file storage fails (F3 related)
        """
        # Validate file format (F1: Unsupported format)
        file_ext = Path(filename).suffix[1:].lower()
        if file_ext not in self.ALLOWED_VIDEO_FORMATS:
            raise InvalidFileFormatException(
                f"File format '{file_ext}' not supported. "
                f"Allowed formats: {', '.join(self.ALLOWED_VIDEO_FORMATS)}"
            )
        
        # Validate file size (F2: File too large)
        if file_size > self.MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            raise FileTooLargeException(
                f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({self.MAX_FILE_SIZE_MB}MB)"
            )
        
        # Store file (Delegation to specialized service - SRP)
        try:
            storage_path, stored_filename = await self._file_storage.save_video(
                file, filename, player_id
            )
        except StorageException:
            raise  # Re-raise storage errors
        
        # Extract video duration from stored file
        full_file_path = self._file_storage.get_file_path(storage_path)
        video_duration = self._extract_video_duration(full_file_path)

        # Create video domain entity
        video = Video(
            id=None,
            file_name=filename,
            storage_path=storage_path,
            status=VideoStatus.UPLOADED,
            upload_timestamp=datetime.now(),
            video_length=video_duration,
            is_deleted=False,
            created_at=None,  # Set by repository
            updated_at=None   # Set by repository
        )
        
        # Persist to database
        created_video = await self._video_repository.create(video)
        
        # TODO: Trigger async analysis queue here (for future implementation)
        # await self._analysis_queue.enqueue(created_video.id)
        
        return created_video
    
    # Internal methods used by analysis service or admin operations
    # Not exposed through public controller but available for internal use
    
    async def delete_video(self, video_id: int) -> bool:
        """
        Soft delete a video
        Used internally by admin operations or cleanup jobs
        
        Returns:
            True if deletion was successful
        """
        return await self._video_repository.soft_delete(video_id)
    
    async def get_video_by_id(self, video_id: int) -> Optional[Video]:
        """
        Get video by ID
        Used internally by analysis service or admin operations
        
        Returns:
            Video entity or None if not found
        """
        return await self._video_repository.get_by_id(video_id)
    
    async def update_video_status(
        self, 
        video_id: int, 
        status: VideoStatus,
        error_message: Optional[str] = None
    ) -> Video:
        """
        Update video processing status
        Implements UC-01 Failure Scenario F4 (Analysis failed)
        Used internally by analysis service
        
        Args:
            video_id: ID of the video
            status: New status to set
            error_message: Optional error message if status is ERROR
            
        Returns:
            Updated video entity
            
        Raises:
            VideoNotFoundException: If video doesn't exist
        """
        video = await self._video_repository.get_by_id(video_id)
        if not video:
            raise VideoNotFoundException(f"Video with ID {video_id} not found")
        
        # Update status
        updated_video = await self._video_repository.update_status(
            video_id, status, error_message
        )
        
        return updated_video
    
    def validate_video_file(self, filename: str, file_size: int) -> tuple[bool, Optional[str]]:
        """
        Validate video file before upload
        Implements UC-01 Failure Scenarios F1 and F2
        
        Business Rules:
        1. File must have allowed extension
        2. File size must not exceed maximum
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file extension (F1: Unsupported format)
        file_extension = Path(filename).suffix[1:].lower()
        if file_extension not in self.ALLOWED_VIDEO_FORMATS:
            return False, f"Format '{file_extension}' not supported. Allowed: {', '.join(self.ALLOWED_VIDEO_FORMATS)}"
        
        # Check file size (F2: File too large)
        if file_size > self.MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            return False, f"File size ({size_mb:.2f}MB) exceeds maximum ({self.MAX_FILE_SIZE_MB}MB)"
        
        return True, None
    
    def get_allowed_formats(self) -> list[str]:
        """Get list of allowed video formats"""
        return self.ALLOWED_VIDEO_FORMATS.copy()
    
    def get_max_file_size_mb(self) -> int:
        """Get maximum allowed file size in MB"""
        return self.MAX_FILE_SIZE_MB
    
    def _extract_video_duration(self, file_path: Path) -> Optional[float]:
        """
        Extract video duration in seconds using ffprobe

        Args:
            file_path: Path to the video file

        Returns:
            Duration in seconds or None if extraction fails
        """
        try: 
            probe = ffmpeg.probe(str(file_path))
            duration = float(probe['format']['duration'])
            return round(duration, 2)
        except Exception as e:
            print(f"Warning: Could not extract video duration: {str(e)}")
            return None
    
