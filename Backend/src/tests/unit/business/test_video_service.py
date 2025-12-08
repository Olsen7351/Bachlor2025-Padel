import pytest
from unittest.mock import Mock, AsyncMock
from io import BytesIO
from datetime import datetime

from app.domain.video import Video, VideoStatus
from app.business.services.video_service import VideoService
from app.business.exceptions import (
    InvalidFileFormatException,
    FileTooLargeException,
    StorageException,
    VideoNotFoundException
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
        return storage
    
    @pytest.fixture
    def video_service(self, mock_repository, mock_storage):
        """Create video service with mocked dependencies"""
        return VideoService(mock_repository, mock_storage)
    
    # Validation Tests
    
    def test_validate_video_file_valid_format(self, video_service):
        """
        Test validation accepts valid video formats
        GIVEN valid video file formats
        WHEN validating the file
        THEN validation passes
        """
        valid_formats = ['test.mp4', 'video.avi', 'match.mov', 'game.mkv', 'clip.webm']
        
        for filename in valid_formats:
            is_valid, error = video_service.validate_video_file(filename, 100 * 1024 * 1024)
            assert is_valid is True
            assert error is None
    
    def test_validate_video_file_invalid_format(self, video_service):
        """
        Test validation rejects invalid formats
        GIVEN an invalid file format
        WHEN validating the file
        THEN validation fails with appropriate error
        """
        is_valid, error = video_service.validate_video_file("video.xyz", 100 * 1024 * 1024)
        
        assert is_valid is False
        assert "not supported" in error.lower()
        assert "xyz" in error
    
    def test_validate_video_file_too_large(self, video_service):
        """
        Test validation rejects oversized files
        GIVEN a file larger than the maximum allowed size
        WHEN validating the file
        THEN validation fails with size error
        """
        from app.config import get_settings
        settings = get_settings()
        
        # Test with file larger than configured limit
        oversized_file_mb = settings.video_max_file_size_mb + 100  # 2100MB
        oversized_file_bytes = oversized_file_mb * 1024 * 1024
        
        is_valid, error = video_service.validate_video_file("video.mp4", oversized_file_bytes)
        
        assert is_valid is False
        assert "exceeds" in error.lower()
    
    def test_validate_video_file_under_limit(self, video_service):
        """
        Test validation accepts files under the size limit
        GIVEN a file under the maximum allowed size
        WHEN validating the file
        THEN validation passes
        """
        from app.config import get_settings
        settings = get_settings()
        
        # Test with file well under limit (1000MB with 2000MB limit)
        acceptable_file_mb = settings.video_max_file_size_mb - 1000  # 1000MB
        acceptable_file_bytes = acceptable_file_mb * 1024 * 1024
        
        is_valid, error = video_service.validate_video_file("video.mp4", acceptable_file_bytes)
        
        assert is_valid is True
        assert error is None
    
    def test_validate_video_file_exactly_at_limit(self, video_service):
        """
        Test validation with file exactly at size limit
        GIVEN a file exactly at the maximum allowed size
        WHEN validating the file
        THEN validation passes
        """
        from app.config import get_settings
        settings = get_settings()
        
        exact_limit_bytes = settings.video_max_file_size_bytes
        
        is_valid, error = video_service.validate_video_file("video.mp4", exact_limit_bytes)
        
        assert is_valid is True
        assert error is None
    
    # Upload Tests
    
    @pytest.mark.asyncio
    async def test_upload_video_success(
        self,
        video_service,
        mock_repository,
        mock_storage
    ):
        """
        UC-01 S1: Successful video upload (Business Logic)
        GIVEN a valid video file
        WHEN uploading the video
        THEN the file is stored and database record is created
        """
        # Arrange
        mock_storage.save_video.return_value = ("path/to/video.mp4", "stored_video.mp4")
        
        created_video = Video(
            id=1,
            file_name="test.mp4",
            storage_path="path/to/video.mp4",
            status=VideoStatus.UPLOADED,
            upload_timestamp=datetime.now(),
            video_length=None,
            is_deleted=False,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        mock_repository.create.return_value = created_video
        
        # Act
        file_content = BytesIO(b"test content")
        result = await video_service.upload_video(
            file=file_content,
            filename="test.mp4",
            content_type="video/mp4",
            file_size=1024,
            player_id="player-123"
        )
        
        # Assert
        assert result.id == 1
        assert result.status == VideoStatus.UPLOADED
        assert result.file_name == "test.mp4"
        
        # Verify storage was called correctly
        mock_storage.save_video.assert_called_once()
        storage_call_args = mock_storage.save_video.call_args
        assert storage_call_args[0][1] == "test.mp4"
        assert storage_call_args[0][2] == "player-123"
        
        # Verify repository was called
        mock_repository.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_video_invalid_format_raises_exception(
        self,
        video_service,
        mock_storage
    ):
        """
        UC-01 F1: Invalid file format (Business Logic)
        GIVEN a file with unsupported format
        WHEN uploading the video
        THEN InvalidFileFormatException is raised
        """
        file_content = BytesIO(b"test content")
        
        with pytest.raises(InvalidFileFormatException) as exc_info:
            await video_service.upload_video(
                file=file_content,
                filename="test.xyz",
                content_type="video/xyz",
                file_size=1024,
                player_id="player-123"
            )
        
        assert "not supported" in str(exc_info.value).lower()
        assert "xyz" in str(exc_info.value)
        mock_storage.save_video.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_upload_video_file_too_large_raises_exception(
        self,
        video_service,
        mock_storage
    ):
        """
        UC-01 F2: File too large (Business Logic)
        GIVEN a file exceeding size limit
        WHEN uploading the video
        THEN FileTooLargeException is raised
        """
        from app.config import get_settings
        settings = get_settings()
        
        oversized_bytes = (settings.video_max_file_size_mb + 100) * 1024 * 1024
        file_content = BytesIO(b"test content")
        
        with pytest.raises(FileTooLargeException) as exc_info:
            await video_service.upload_video(
                file=file_content,
                filename="test.mp4",
                content_type="video/mp4",
                file_size=oversized_bytes,
                player_id="player-123"
            )
        
        assert "exceeds" in str(exc_info.value).lower()
        mock_storage.save_video.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_upload_video_storage_error_propagates(
        self,
        video_service,
        mock_storage
    ):
        """
        UC-01 F3: Storage error (Business Logic)
        GIVEN a valid file
        WHEN storage fails
        THEN StorageException is propagated
        """
        # Arrange
        mock_storage.save_video.side_effect = StorageException("Disk full")
        file_content = BytesIO(b"test content")
        
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
    
    # Internal Service Methods Tests
    
    @pytest.mark.asyncio
    async def test_get_video_by_id_success(
        self,
        video_service,
        mock_repository
    ):
        """
        Test retrieving video by ID
        GIVEN a valid video ID
        WHEN getting video by ID
        THEN the video entity is returned
        """
        # Arrange
        expected_video = Video(
            id=1,
            file_name="test.mp4",
            storage_path="path/to/video.mp4",
            status=VideoStatus.UPLOADED,
            upload_timestamp=datetime.now(),
            video_length=None,
            is_deleted=False
        )
        mock_repository.get_by_id.return_value = expected_video
        
        # Act
        result = await video_service.get_video_by_id(1)
        
        # Assert
        assert result.id == 1
        assert result.file_name == "test.mp4"
        mock_repository.get_by_id.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_get_video_by_id_not_found(
        self,
        video_service,
        mock_repository
    ):
        """
        Test retrieving non-existent video
        GIVEN an invalid video ID
        WHEN getting video by ID
        THEN None is returned
        """
        # Arrange
        mock_repository.get_by_id.return_value = None
        
        # Act
        result = await video_service.get_video_by_id(999)
        
        # Assert
        assert result is None
        mock_repository.get_by_id.assert_called_once_with(999)
    
    @pytest.mark.asyncio
    async def test_update_video_status_success(
        self,
        video_service,
        mock_repository
    ):
        """
        Test updating video status
        GIVEN a valid video ID and new status
        WHEN updating video status
        THEN the video is updated successfully
        """
        # Arrange
        existing_video = Video(
            id=1,
            file_name="test.mp4",
            storage_path="path/to/video.mp4",
            status=VideoStatus.UPLOADED,
            upload_timestamp=datetime.now(),
            video_length=None,
            is_deleted=False
        )
        updated_video = Video(
            id=1,
            file_name="test.mp4",
            storage_path="path/to/video.mp4",
            status=VideoStatus.ANALYZED,
            upload_timestamp=datetime.now(),
            video_length=None,
            is_deleted=False
        )
        
        mock_repository.get_by_id.return_value = existing_video
        mock_repository.update_status.return_value = updated_video
        
        # Act
        result = await video_service.update_video_status(1, VideoStatus.ANALYZED)
        
        # Assert
        assert result.status == VideoStatus.ANALYZED
        mock_repository.get_by_id.assert_called_once_with(1)
        mock_repository.update_status.assert_called_once_with(1, VideoStatus.ANALYZED, None)
    
    @pytest.mark.asyncio
    async def test_update_video_status_not_found(
        self,
        video_service,
        mock_repository
    ):
        """
        Test updating status of non-existent video
        GIVEN an invalid video ID
        WHEN updating video status
        THEN VideoNotFoundException is raised
        """
        # Arrange
        mock_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(VideoNotFoundException) as exc_info:
            await video_service.update_video_status(999, VideoStatus.ANALYZED)
        
        assert "not found" in str(exc_info.value).lower()
        mock_repository.update_status.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_delete_video_success(
        self,
        video_service,
        mock_repository
    ):
        """
        Test soft deleting a video
        GIVEN a valid video ID
        WHEN deleting the video
        THEN the video is soft deleted
        """
        # Arrange
        mock_repository.soft_delete.return_value = True
        
        # Act
        result = await video_service.delete_video(1)
        
        # Assert
        assert result is True
        mock_repository.soft_delete.assert_called_once_with(1)
    
    # Configuration Tests
    
    def test_get_allowed_formats(self, video_service):
        """
        Test getting allowed formats
        WHEN requesting allowed formats
        THEN a list of valid formats is returned
        """
        formats = video_service.get_allowed_formats()
        
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert 'mp4' in formats
        assert 'avi' in formats
    
    def test_get_max_file_size_mb(self, video_service):
        """
        Test getting maximum file size
        WHEN requesting max file size
        THEN the configured value is returned
        """
        from app.config import get_settings
        settings = get_settings()
        
        max_size = video_service.get_max_file_size_mb()
        
        assert isinstance(max_size, int)
        assert max_size == settings.video_max_file_size_mb
        assert max_size > 0


# Run tests with: pytest tests/unit/business/test_video_service.py -v