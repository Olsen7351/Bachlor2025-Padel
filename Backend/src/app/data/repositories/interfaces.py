from abc import abstractmethod
from typing import Optional, List
from .base_repository import BaseRepository

from ...domain.player import Player
from ..models.player_model import PlayerModel

from ...domain.video import Video, VideoStatus
from ..models.video_model import VideoModel

from ...domain.analysis import Analysis
from ..models.analysis_model import AnalysisModel

from ...domain.match import Match, MatchPlayer, SummaryMetrics
from ..models.match_model import MatchModel, MatchPlayerModel, SummaryMetricsModel

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

class IAnalysisRepository(BaseRepository[Analysis, AnalysisModel]):
    """
    Analysis-specific repository interface
    """
    @abstractmethod
    async def get_by_video_id(self, video_id: int) -> Optional[Analysis]:
        """Get analysis by video ID - analysis-specific query"""
        pass

    @abstractmethod
    async def get_by_player_id(self, player_id: str) -> List[Analysis]:
        """Get all analyses for a player - analysis-specific query"""
        pass

class IMatchRepository(BaseRepository[Match, MatchModel]):
    """Match-specific repository interface"""
    @abstractmethod
    async def get_by_analysis_id(self, anaysis_id: int) -> Optional[Match]:
        """Get match by analysis ID - match-specific query"""
        pass

    @abstractmethod
    async def get_match_with_players(self, match_id: int) -> Optional[Match]:
        """Get match with all associated players - eager loading"""
        pass

class IMatchPlayerRepository(BaseRepository[MatchPlayer, MatchPlayerModel]):
    """MatchPlayer-specific repository interface"""
    @abstractmethod
    async def get_by_match_id(self, match_id: int) -> List[MatchPlayer]:
        """Get all players for a specific match"""
        pass

    @abstractmethod
    async def get_by_identifier(self, match_id: int, player_identifier: str) -> Optional[MatchPlayer]:
        """Get a specific player by their identifier in a match"""
        pass

class ISummaryMetricsRepository(BaseRepository[SummaryMetrics, SummaryMetricsModel]):
    """SummaryMetrics-specific repository interface"""
    @abstractmethod
    async def get_by_match_player_id(self, match_player_id: int) -> Optional[SummaryMetrics]:
        """Get summary metrics for a specific match player"""
        pass

    @abstractmethod
    async def get_all_by_match_id(self, match_id: int) -> List[SummaryMetrics]:
        """Get summary metrics for all players in a match - UC-04 S1"""
        pass

    @abstractmethod
    async def get_by_match_and_set(self, match_id: int, set_number: int) -> List[SummaryMetrics]:
        """Get summary metrics filtered by set number - UC-04 S2"""
        pass
    