from abc import ABC, abstractmethod
from typing import Optional, BinaryIO, Tuple
from pathlib import Path
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


class IFileStorageService(ABC):
    """
    Interface for file storage operations
    Following Dependency Inversion Principle
    """
    
    @abstractmethod
    async def save_video(
        self, 
        file: BinaryIO, 
        original_filename: str, 
        player_id: str
    ) -> Tuple[str, str]:
        """
        Save video file to storage
        
        Returns:
            Tuple of (storage_path, stored_filename)
        """
        pass
    
    @abstractmethod
    async def delete_video(self, storage_path: str) -> bool:
        """Delete video file from storage"""
        pass
    
    @abstractmethod
    def get_file_path(self, storage_path: str) -> Path:
        """Get absolute path for a stored file"""
        pass
    
    @abstractmethod
    def file_exists(self, storage_path: str) -> bool:
        """Check if file exists in storage"""
        pass