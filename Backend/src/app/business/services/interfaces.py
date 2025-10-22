from abc import ABC, abstractmethod
from typing import Optional, BinaryIO
from ...domain.player import Player
from ...domain.video import Video, VideoStatus


class IPlayerService(ABC):
    """
    Interface for Player business logic
    """
    
    @abstractmethod
    async def create_player(self, id: str, name: str, email: str, role: str = "player") -> Player:
        """Create a new player with validation"""
        pass
    
    @abstractmethod
    async def get_player_by_id(self, player_id: str) -> Player:
        """Get player by ID"""
        pass
    
    @abstractmethod
    async def get_player_by_email(self, email: str) -> Player:
        """Get player by email"""
        pass
    


class IVideoService(ABC):
    """Interface for Video service following Service Layer pattern"""
    
    @abstractmethod
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
        
        Args:
            file: The video file binary content
            filename: Original filename
            content_type: MIME type of the file
            file_size: Size of the file in bytes
            player_id: ID of the player uploading the video
            
        Returns:
            Video domain entity
            
        Raises:
            InvalidFileFormatError: If file format is not supported
            FileTooLargeError: If file exceeds size limit
            StorageError: If file storage fails
        """
        pass
    
    @abstractmethod
    async def get_video_by_id(self, video_id: int) -> Optional[Video]:
        """Get video by ID"""
        pass
    
    @abstractmethod
    async def update_video_status(
        self, 
        video_id: int, 
        status: VideoStatus,
        error_message: Optional[str] = None
    ) -> Video:
        """Update video processing status"""
        pass
    
    @abstractmethod
    def validate_video_file(self, filename: str, file_size: int) -> tuple[bool, Optional[str]]:
        """
        Validate video file before upload
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    async def delete_video(self, video_id: int) -> bool:
        """Soft delete a video"""
        pass