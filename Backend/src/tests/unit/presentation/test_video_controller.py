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
        # Mock get_video_path needed for the background task arg preparation
        service.get_video_path = Mock(return_value="/tmp/some/path.mp4")
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
        content = b"fake video content" * 1000 
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
        """
        # Arrange
        mock_video_service.upload_video.return_value = created_video
        
        from app.presentation.controllers.video_controller import upload_video
        
        # Act
        response = await upload_video(
            file=valid_video_file,
            background_tasks=mock_background_tasks,
            court_number=9,  # ADDED ARGUMENT
            current_user=mock_player,
            video_service=mock_video_service,
            session=mock_session
        )
        
        # Assert
        assert response.id == 1
        assert response.status == "uploaded"
        
        # Verify transaction was committed BEFORE background task
        mock_session.commit.assert_called_once()
        
        # Verify background task was scheduled
        mock_background_tasks.add_task.assert_called_once()
        task_call = mock_background_tasks.add_task.call_args
        assert task_call.kwargs['court_number'] == 9 # Check court number passed to task
    
    @pytest.mark.asyncio
    async def test_upload_video_no_file_provided(
        self,
        mock_player,
        mock_video_service,
        mock_background_tasks,
        mock_session
    ):
        from app.presentation.controllers.video_controller import upload_video
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await upload_video(
                file=None,
                background_tasks=mock_background_tasks,
                court_number=9, # ADDED ARGUMENT
                current_user=mock_player,
                video_service=mock_video_service,
                session=mock_session
            )
        
        assert exc_info.value.status_code == 400
        assert "no file" in str(exc_info.value.detail).lower()
    
    # F1: Unsupported file format
    @pytest.mark.asyncio
    async def test_upload_video_invalid_format(
        self,
        mock_player,
        mock_video_service,
        mock_background_tasks,
        mock_session
    ):
        # Arrange
        invalid_file = UploadFile(
            filename="test_video.xyz",
            file=BytesIO(b"content")
        )
        
        mock_video_service.upload_video.side_effect = InvalidFileFormatException(
            "File format 'xyz' not supported."
        )
        
        from app.presentation.controllers.video_controller import upload_video
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await upload_video(
                file=invalid_file,
                background_tasks=mock_background_tasks,
                court_number=9, # ADDED ARGUMENT
                current_user=mock_player,
                video_service=mock_video_service,
                session=mock_session
            )
        
        assert exc_info.value.status_code == 400
    
    # F2: File too large
    @pytest.mark.asyncio
    async def test_upload_video_file_too_large(
        self,
        mock_player,
        mock_video_service,
        mock_background_tasks,
        mock_session
    ):
        # Arrange
        large_file = UploadFile(
            filename="large_video.mp4",
            file=BytesIO(b"x" * 1000)
        )
        
        mock_video_service.upload_video.side_effect = FileTooLargeException(
            "File size exceeds maximum"
        )
        
        from app.presentation.controllers.video_controller import upload_video
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await upload_video(
                file=large_file,
                background_tasks=mock_background_tasks,
                court_number=9, # ADDED ARGUMENT
                current_user=mock_player,
                video_service=mock_video_service,
                session=mock_session
            )
        
        assert exc_info.value.status_code == 400
    
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
                court_number=9, # ADDED ARGUMENT
                current_user=mock_player,
                video_service=mock_video_service,
                session=mock_session
            )
        
        assert exc_info.value.status_code == 500
    
    @pytest.mark.asyncio
    async def test_upload_video_unexpected_error(
        self,
        mock_player,
        mock_video_service,
        mock_background_tasks,
        mock_session,
        valid_video_file
    ):
        # Arrange
        mock_video_service.upload_video.side_effect = Exception("Unexpected error")
        
        from app.presentation.controllers.video_controller import upload_video
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await upload_video(
                file=valid_video_file,
                background_tasks=mock_background_tasks,
                court_number=9, # ADDED ARGUMENT
                current_user=mock_player,
                video_service=mock_video_service,
                session=mock_session
            )
        
        assert exc_info.value.status_code == 500
        mock_session.rollback.assert_called_once()