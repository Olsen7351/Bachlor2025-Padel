from typing import BinaryIO, Optional
from pathlib import Path
from datetime import datetime
import tempfile
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
from .video_converter import VideoConverter


class VideoService(IVideoService):
    """
    Video service implementation.
    Handles business logic for video operations including conversion.
    """
    
    def __init__(
        self, 
        video_repository: IVideoRepository,
        file_storage_service: IFileStorageService,
        video_converter: Optional[VideoConverter] = None
    ):
        """
        Initialize service with dependencies (Dependency Inversion Principle)
        
        Args:
            video_repository: Repository interface for video data access
            file_storage_service: Storage service interface for file operations
            video_converter: Video converter for normalizing uploads
        """
        self._video_repository = video_repository
        self._file_storage = file_storage_service
        self._video_converter = video_converter or VideoConverter()
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
        player_id: str,
        court_number: Optional[int] = None
    ) -> Video:
        """
        Upload and process a video file.
        
        Business Rules:
        1. Validate file format (F1)
        2. Validate file size (F2)
        3. Convert video if needed (>1080p or >30fps)
        4. Store file securely
        5. Create database record
        6. Set initial status to UPLOADED
        
        Args:
            file: Binary file content
            filename: Original filename
            content_type: MIME type
            file_size: File size in bytes
            player_id: ID of uploading player
            court_number: Optional court number (stored for later analysis)
            
        Returns:
            Video domain entity with UPLOADED status
            
        Raises:
            InvalidFileFormatException: If file format is not supported (F1)
            FileTooLargeException: If file exceeds size limit (F2)
            StorageException: If file storage fails (F3)
        """
        # Validate file format
        file_ext = Path(filename).suffix[1:].lower()
        if file_ext not in self.ALLOWED_VIDEO_FORMATS:
            raise InvalidFileFormatException(
                f"File format '{file_ext}' not supported. "
                f"Allowed formats: {', '.join(self.ALLOWED_VIDEO_FORMATS)}"
            )
        
        # Validate file size
        if file_size > self.MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            raise FileTooLargeException(
                f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({self.MAX_FILE_SIZE_MB}MB)"
            )
        
        # Save to temp file first for conversion check
        temp_path = None
        converted_path = None
        
        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(file.read())
            
            # Check if conversion is needed and convert
            final_path = await self._video_converter.convert_async(temp_path)
            
            # If converted, update the filename
            if final_path != temp_path:
                converted_path = final_path
                filename = f"{Path(filename).stem}_converted.mp4"
            
            # Read the final file for storage
            with open(final_path, 'rb') as f:
                file_content = f
                
                # Store file
                storage_path, stored_filename = await self._file_storage.save_video(
                    f, filename, player_id
                )
            
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
                created_at=None,
                updated_at=None
            )
            
            # Persist to database
            created_video = await self._video_repository.create(video)
            
            return created_video
            
        except (InvalidFileFormatException, FileTooLargeException, StorageException):
            raise
        except Exception as e:
            raise StorageException(f"Failed to process video: {str(e)}")
        finally:
            # Cleanup temp files
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            if converted_path and converted_path.exists():
                try:
                    converted_path.unlink()
                except:
                    pass
    
    async def delete_video(self, video_id: int) -> bool:
        """Soft delete a video"""
        return await self._video_repository.soft_delete(video_id)
    
    def get_allowed_formats(self) -> list[str]:
        """Get list of allowed video formats"""
        return list(self.ALLOWED_VIDEO_FORMATS)
    
    def get_max_file_size_mb(self) -> int:
        """Get maximum allowed file size in MB"""
        return self.MAX_FILE_SIZE_MB
    
    def get_video_path(self, video: Video) -> Path:
        """Get the full path to a video file"""
        return self._file_storage.get_file_path(video.storage_path)
    
    def _extract_video_duration(self, file_path: Path) -> Optional[float]:
        """Extract video duration in seconds using ffprobe"""
        try: 
            probe = ffmpeg.probe(str(file_path))
            duration = float(probe['format']['duration'])
            return round(duration, 2)
        except Exception as e:
            print(f"Warning: Could not extract video duration: {str(e)}")
            return None
