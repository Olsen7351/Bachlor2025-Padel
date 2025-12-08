import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import UploadFile, HTTPException
from starlette.background import BackgroundTasks
from io import BytesIO
from datetime import datetime

from app.domain.video import Video, VideoStatus
from app.domain.player import Player
from app.business.exceptions import (
    InvalidFileFormatException,
    FileTooLargeException,
    StorageException
)


class TestVideoController:
    """Test cases for Video Controller endpoints (Presentation Layer)"""
    
    @pytest.fixture
    def mock_player(self):
        """Mock authenticated player"""
        return Player(
            id="test-player-123",
            name="Test Player",
            email="test@example.com",
            role="player"
        )
    
    @pytest.fixture
    def mock_video_service(self):
        """Mock video service"""
        service = Mock()
        service.upload_video = AsyncMock()
        service.get_allowed_formats = Mock(return_value=['mp4', 'avi', 'mov', 'mkv', 'webm'])
        service.get_max_file_size_mb = Mock(return_value=2000)
        return service
    
    @pytest.fixture
    def mock_background_tasks(self):
        """Mock FastAPI BackgroundTasks"""
        tasks = Mock(spec=BackgroundTasks)
        tasks.add_task = Mock()
        return tasks
    
    @pytest.fixture
    def mock_session(self):
        """Mock SQLAlchemy AsyncSession"""
        session = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session
    
    @pytest.fixture
    def valid_video_file(self):
        """Create a mock valid video file"""
        content = b"fake video content" * 1000  # ~18KB
        file = BytesIO(content)
        upload_file = UploadFile(
            filename="test_match.mp4",
            file=file
        )
        return upload_file
    
    @pytest.fixture
    def created_video(self):
        """Mock created video entity"""
        return Video(
            id=1,
            file_name="test_match.mp4",
            storage_path="test-player-123/20240101_120000_abc123.mp4",
            status=VideoStatus.UPLOADED,
            upload_timestamp=datetime.now(),
            video_length=25.5,
            is_deleted=False,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    # S1: Successful upload
    @pytest.mark.asyncio
    async def test_upload_video_success(
        self, 
        mock_player, 
        mock_video_service, 
        mock_background_tasks,
        mock_session,
        valid_video_file,
        created_video
    ):
        """
        UC-01 S1: Successful video upload (Controller Layer)
        GIVEN a logged-in user with a valid video file
        WHEN they call the upload endpoint
        THEN the system returns 201 with video details
        AND background task is scheduled
        AND transaction is committed
        """
        # Arrange
        mock_video_service.upload_video.return_value = created_video
        
        from app.presentation.controllers.video_controller import upload_video
        
        # Act
        response = await upload_video(
            file=valid_video_file,
            background_tasks=mock_background_tasks,
            current_user=mock_player,
            video_service=mock_video_service,
            session=mock_session
        )
        
        # Assert - Verify DTO mapping and response structure
        assert response.id == 1
        assert response.file_name == "test_match.mp4"
        assert response.status == "uploaded"
        assert response.video_length == 25.5
        assert "successfully" in response.message.lower()
        
        # Verify service was called with correct parameters
        call_args = mock_video_service.upload_video.call_args
        assert call_args.kwargs['filename'] == "test_match.mp4"
        assert call_args.kwargs['player_id'] == "test-player-123"
        mock_video_service.upload_video.assert_called_once()
        
        # Verify transaction was committed BEFORE background task
        mock_session.commit.assert_called_once()
        
        # Verify background task was scheduled
        mock_background_tasks.add_task.assert_called_once()
        # Verify it's the correct task
        task_call = mock_background_tasks.add_task.call_args
        assert task_call.args[0].__name__ == 'process_video_analysis'
        assert task_call.kwargs['video_id'] == 1
        assert task_call.kwargs['player_id'] == "test-player-123"
    
    @pytest.mark.asyncio
    async def test_upload_video_no_file_provided(
        self,
        mock_player,
        mock_video_service,
        mock_background_tasks,
        mock_session
    ):
        """
        Test that endpoint rejects request with no file
        GIVEN a request with no file
        WHEN the upload endpoint is called
        THEN it returns 400 Bad Request
        """
        from app.presentation.controllers.video_controller import upload_video
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await upload_video(
                file=None,
                background_tasks=mock_background_tasks,
                current_user=mock_player,
                video_service=mock_video_service,
                session=mock_session
            )
        
        assert exc_info.value.status_code == 400
        assert "no file" in str(exc_info.value.detail).lower()
        
        # Verify nothing was committed or scheduled
        mock_session.commit.assert_not_called()
        mock_background_tasks.add_task.assert_not_called()
    
    # F1: Unsupported file format
    @pytest.mark.asyncio
    async def test_upload_video_invalid_format(
        self,
        mock_player,
        mock_video_service,
        mock_background_tasks,
        mock_session
    ):
        """
        UC-01 F1: File format not supported (Controller Layer)
        GIVEN a user with an unsupported file format
        WHEN they call the upload endpoint
        THEN the system returns 400 with format error details
        """
        # Arrange
        invalid_file = UploadFile(
            filename="test_video.xyz",
            file=BytesIO(b"content")
        )
        
        mock_video_service.upload_video.side_effect = InvalidFileFormatException(
            "File format 'xyz' not supported. Allowed formats: mp4, avi, mov"
        )
        
        from app.presentation.controllers.video_controller import upload_video
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await upload_video(
                file=invalid_file,
                background_tasks=mock_background_tasks,
                current_user=mock_player,
                video_service=mock_video_service,
                session=mock_session
            )
        
        assert exc_info.value.status_code == 400
        detail = exc_info.value.detail
        assert "error" in detail
        assert "supported_formats" in detail
        assert "max_size_mb" in detail
        
        # Verify transaction was not committed
        mock_session.commit.assert_not_called()
        mock_background_tasks.add_task.assert_not_called()
    
    # F2: File too large
    @pytest.mark.asyncio
    async def test_upload_video_file_too_large(
        self,
        mock_player,
        mock_video_service,
        mock_background_tasks,
        mock_session
    ):
        """
        UC-01 F2: File exceeds size limit (Controller Layer)
        GIVEN a user with a file exceeding the size limit
        WHEN they call the upload endpoint
        THEN the system returns 400 with size error details
        """
        # Arrange
        large_file = UploadFile(
            filename="large_video.mp4",
            file=BytesIO(b"x" * 1000)
        )
        
        mock_video_service.upload_video.side_effect = FileTooLargeException(
            "File size (2100.00MB) exceeds maximum allowed size (2000MB)"
        )
        
        from app.presentation.controllers.video_controller import upload_video
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await upload_video(
                file=large_file,
                background_tasks=mock_background_tasks,
                current_user=mock_player,
                video_service=mock_video_service,
                session=mock_session
            )
        
        assert exc_info.value.status_code == 400
        detail = exc_info.value.detail
        assert "error" in detail
        assert "max_size_mb" in detail
        
        # Verify transaction was not committed
        mock_session.commit.assert_not_called()
        mock_background_tasks.add_task.assert_not_called()
    
    # F3: Storage error
    @pytest.mark.asyncio
    async def test_upload_video_storage_error(
        self,
        mock_player,
        mock_video_service,
        mock_background_tasks,
        mock_session,
        valid_video_file
    ):
        """
        UC-01 F3: Network/storage error during upload (Controller Layer)
        GIVEN a user uploading a valid file
        WHEN storage operation fails
        THEN the system returns 500 with storage error
        """
        # Arrange
        mock_video_service.upload_video.side_effect = StorageException(
            "Failed to save file: Disk full"
        )
        
        from app.presentation.controllers.video_controller import upload_video
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await upload_video(
                file=valid_video_file,
                background_tasks=mock_background_tasks,
                current_user=mock_player,
                video_service=mock_video_service,
                session=mock_session
            )
        
        assert exc_info.value.status_code == 500
        detail = exc_info.value.detail
        assert "error" in detail
        assert "failed to store" in str(detail).lower()
        
        # Verify transaction was not committed
        mock_session.commit.assert_not_called()
        mock_background_tasks.add_task.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_upload_video_unexpected_error(
        self,
        mock_player,
        mock_video_service,
        mock_background_tasks,
        mock_session,
        valid_video_file
    ):
        """
        Test handling of unexpected exceptions
        GIVEN a user uploading a file
        WHEN an unexpected error occurs
        THEN the system returns 500 with generic error
        AND transaction is rolled back
        """
        # Arrange
        mock_video_service.upload_video.side_effect = Exception("Unexpected error")
        
        from app.presentation.controllers.video_controller import upload_video
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await upload_video(
                file=valid_video_file,
                background_tasks=mock_background_tasks,
                current_user=mock_player,
                video_service=mock_video_service,
                session=mock_session
            )
        
        assert exc_info.value.status_code == 500
        detail = exc_info.value.detail
        assert "unexpected error" in str(detail).lower()
        
        # Verify rollback was called
        mock_session.rollback.assert_called_once()
        
        # Verify no commit or background task
        mock_session.commit.assert_not_called()
        mock_background_tasks.add_task.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_upload_config(self, mock_video_service):
        """
        Test upload configuration endpoint
        GIVEN a request for upload configuration
        WHEN the config endpoint is called
        THEN it returns max file size and allowed formats
        """
        from app.presentation.controllers.video_controller import get_upload_config
        
        # Act
        response = await get_upload_config(video_service=mock_video_service)
        
        # Assert
        assert "max_file_size_mb" in response
        assert "allowed_formats" in response
        assert "description" in response
        assert response["max_file_size_mb"] == 2000
        assert response["allowed_formats"] == ['mp4', 'avi', 'mov', 'mkv', 'webm']

