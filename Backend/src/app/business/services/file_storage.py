import os
import aiofiles
from pathlib import Path
from typing import BinaryIO
from datetime import datetime
import uuid

from ...business.exceptions import StorageException
from ...business.services.interfaces import IFileStorageService
from ...config import get_settings


class FileStorageService(IFileStorageService):
    """Service for handling file storage operations - follows Single Responsibility Principle"""
    
    def __init__(self, base_storage_path: str = None):
        """
        Initialize file storage service
        
        Args:
            base_storage_path: Optional custom path. If None, uses config setting.
        """
        settings = get_settings()
        self.base_storage_path = Path(base_storage_path or settings.video_upload_dir)
        self._ensure_storage_directory()
    
    def _ensure_storage_directory(self):
        """Create storage directory if it doesn't exist"""
        try:
            self.base_storage_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise StorageException(f"Failed to create storage directory: {str(e)}")
    
    async def save_video(self, file: BinaryIO, original_filename: str, player_id: str) -> tuple[str, str]:
        """
        Save video file to storage
        
        Args:
            file: Binary file content
            original_filename: Original name of the file
            player_id: ID of the player (for organizing files)
            
        Returns:
            Tuple of (storage_path, stored_filename)
            
        Raises:
            StorageError: If file save operation fails
        """
        try:
            # Create player-specific directory
            player_dir = self.base_storage_path / player_id
            player_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename to avoid conflicts
            file_extension = Path(original_filename).suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            stored_filename = f"{timestamp}_{unique_id}{file_extension}"
            
            # Full path for the file
            file_path = player_dir / stored_filename
            
            # Read content sync
            content = file.read()

            # Write file async
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # Return relative path from base storage and the stored filename
            relative_path = str(file_path.relative_to(self.base_storage_path))
            
            return relative_path, stored_filename
            
        except Exception as e:
            raise StorageException(f"Failed to save file {original_filename}: {str(e)}")
    
    async def delete_video(self, storage_path: str) -> bool:
        """
        Delete video file from storage
        
        Args:
            storage_path: Relative path to the file
            
        Returns:
            True if deletion was successful
            
        Raises:
            StorageError: If file deletion fails
        """
        try:
            file_path = self.base_storage_path / storage_path
            
            if file_path.exists():
                file_path.unlink()
                return True
            return False
            
        except Exception as e:
            raise StorageException(f"Failed to delete file {storage_path}: {str(e)}")
    
    def get_file_path(self, storage_path: str) -> Path:
        """
        Get absolute path for a stored file
        
        Args:
            storage_path: Relative path to the file
            
        Returns:
            Absolute Path object
        """
        return self.base_storage_path / storage_path
    
    def file_exists(self, storage_path: str) -> bool:
        """Check if file exists in storage"""
        file_path = self.base_storage_path / storage_path
        return file_path.exists()