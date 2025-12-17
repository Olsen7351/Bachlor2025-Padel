from abc import abstractmethod, ABC
from typing import Optional, List
from .base_repository import BaseRepository

from ...domain.player import Player
from ..models.player_model import PlayerModel

from ...domain.video import Video, VideoStatus
from ..models.video_model import VideoModel

from ...domain.analysis import Analysis
from ..models.analysis_model import AnalysisModel

from ...domain.match import Match, MatchPlayer, SummaryMetrics, Heatmap, Rally, Hits
from ..models.match_model import MatchModel, MatchPlayerModel, SummaryMetricsModel, HeatmapModel, RallyModel, HitsModel

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
    async def soft_delete(self, video_id: int) -> bool:
        """Soft delete video (set is_deleted=True) - video-specific operation"""
        pass

    @abstractmethod
    async def get_analyzed_by_player_id(self, player_id: str) -> List[Video]:
        """Get all analyzed videos uploaded by a specific player"""
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
    async def get_by_match_id(self, match_id: int) -> Optional[Analysis]:
        """Get analysis by match ID - analysis-specific query"""
        pass

class IMatchRepository(BaseRepository[Match, MatchModel]):
    """Match-specific repository interface"""
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

    
class IHitsRepository(BaseRepository[Hits, HitsModel], ABC):
    """Hits repository interface - standalone entity"""
    pass


class IRallyRepository(BaseRepository[Rally, RallyModel], ABC):
    """
    Rally repository interface - standalone entity
    """
    
    @abstractmethod
    async def get_by_summary_metrics_id(self, summary_metrics_id: int) -> List[Rally]:
        """Get all rallies for a summary metrics record"""
        pass


class IHeatmapRepository(BaseRepository[Heatmap, HeatmapModel], ABC):
    """Heatmap repository interface - standalone entity"""
    pass
