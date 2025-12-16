import pytest
from unittest.mock import Mock, AsyncMock, patch
from io import BytesIO
from datetime import datetime
from pathlib import Path

from app.domain.video import Video, VideoStatus
from app.business.services.video_service import VideoService
from app.business.exceptions import (
    InvalidFileFormatException,
    FileTooLargeException,
    StorageException
)

class TestVideoService:
    """Test cases for Video Service business logic"""
    
    @pytest.fixture
    def mock_repository(self):
        """Mock video repository"""
        repo = Mock()
        repo.create = AsyncMock()
        repo.get_by_id = AsyncMock()
        repo.update_status = AsyncMock()
        repo.soft_delete = AsyncMock()
        return repo
    
    @pytest.fixture
    def mock_storage(self):
        """Mock file storage service"""
        storage = Mock()
        storage.save_video = AsyncMock()
        storage.delete_video = AsyncMock()
        # Return a dummy path when requested
        storage.get_file_path = Mock(return_value=Path("/tmp/dummy_video.mp4"))
        return storage

    @pytest.fixture
    def mock_converter(self):
        """Mock video converter to bypass ffmpeg checks"""
        converter = Mock()
        # Mock convert_async to simply return the input path without processing
        converter.convert_async = AsyncMock(side_effect=lambda p: p)
        return converter
    
    @pytest.fixture
    def video_service(self, mock_repository, mock_storage, mock_converter):
        """Create video service with ALL mocked dependencies"""
        return VideoService(mock_repository, mock_storage, mock_converter)
    
    # --- Upload Tests ---
    
    @pytest.mark.asyncio
    async def test_upload_video_success(
        self,
        video_service,
        mock_repository,
        mock_storage
    ):
        """
        UC-01 S1: Successful video upload
        """
        # Arrange
        mock_storage.save_video.return_value = ("path/to/video.mp4", "stored_video.mp4")
        
        # Ensure Video object has the new 'storage_path' argument
        created_video = Video(
            id=1,
            file_name="test.mp4",
            storage_path="path/to/video.mp4", # <--- Added this
            status=VideoStatus.UPLOADED,
            upload_timestamp=datetime.now(),
            video_length=120.0,
            is_deleted=False,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        mock_repository.create.return_value = created_video
        
        file_content = BytesIO(b"test content")

        # Mock the internal _extract_video_duration method to avoid the second ffprobe call
        with patch.object(video_service, '_extract_video_duration', return_value=120.0):
            # Act
            result = await video_service.upload_video(
                file=file_content,
                filename="test.mp4",
                content_type="video/mp4",
                file_size=1024,
                player_id="player-123",
                court_number=1
            )
        
        # Assert
        assert result.id == 1
        assert result.status == VideoStatus.UPLOADED
        assert result.storage_path == "path/to/video.mp4"
        
        mock_storage.save_video.assert_called_once()
        mock_repository.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_video_storage_error_propagates(
        self,
        video_service,
        mock_storage
    ):
        """
        UC-01 F3: Storage error
        """
        # Arrange
        mock_storage.save_video.side_effect = StorageException("Disk full")
        file_content = BytesIO(b"test content")
        
        # Mock extraction just in case, though it shouldn't be reached if storage fails
        with patch.object(video_service, '_extract_video_duration', return_value=120.0):
            # Act & Assert
            with pytest.raises(StorageException) as exc_info:
                await video_service.upload_video(
                    file=file_content,
                    filename="test.mp4",
                    content_type="video/mp4",
                    file_size=1024,
                    player_id="player-123"
                )
        
        assert "disk full" in str(exc_info.value).lower()
        
    @pytest.mark.asyncio
    async def test_upload_video_invalid_format_raises_exception(
        self,
        video_service,
        mock_storage
    ):
        file_content = BytesIO(b"test content")
        with pytest.raises(InvalidFileFormatException):
            await video_service.upload_video(
                file=file_content,
                filename="test.xyz",
                content_type="video/xyz",
                file_size=1024,
                player_id="123"
            )

    @pytest.mark.asyncio
    async def test_upload_video_file_too_large_raises_exception(
        self,
        video_service,
        mock_storage
    ):
        from app.config import get_settings
        settings = get_settings()
        oversized = (settings.video_max_file_size_mb + 100) * 1024 * 1024
        file_content = BytesIO(b"test content")
        
        with pytest.raises(FileTooLargeException):
            await video_service.upload_video(
                file=file_content,
                filename="test.mp4",
                content_type="video/mp4",
                file_size=oversized,
                player_id="123"
            )