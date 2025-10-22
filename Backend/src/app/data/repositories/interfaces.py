from abc import abstractmethod
from typing import Optional, List
from .base_repository import BaseRepository

from ...domain.player import Player
from ..models.player_model import PlayerModel

from ...domain.video import Video, VideoStatus
from ..models.video_model import VideoModel

class IPlayerRepository(BaseRepository[Player, PlayerModel]):
    """
    Player-specific repository interface
    Extends BaseRepository with player-specific methods
    """
    
    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[Player]:
        """Get player by email - player-specific query"""
        pass
    

class IVideoRepository(BaseRepository[Video, VideoModel]):
    """
    Video-specific repository interface
    Extends BaseRepository with video-specific methods
    """

    @abstractmethod
    async def update_status(
        self,
        video_id: int,
        status: VideoStatus,
        error_message: Optional[str] = None
    ) -> Video:
        """Update video status - video-specific operation"""
        pass

    @abstractmethod
    async def get_by_status(self, status: VideoStatus) -> List[Video]:
        """Get all videos with specific status - video-specific query"""
        pass

    @abstractmethod
    async def soft_delete(self, video_id: int) -> bool:
        """Soft delete video (set is_deleted=True) - video-specific operation"""
        pass