from .interfaces import IHeatmapService
from ..exceptions import (
    MatchNotFoundException,
    PlayerInMatchNotFoundException,
    DataUnavailableException,
    HeatmapNotFoundException
)
from ...data.repositories.interfaces import (
    IMatchRepository,
    IMatchPlayerRepository,
    ISummaryMetricsRepository,
    IHeatmapRepository
)


class HeatmapService(IHeatmapService):
    """
    Service for heatmap visualization business logic.
    
    Responsibilities:
    - Retrieve and encode heatmaps for API responses
    - Handle heatmap comparison logic
    - Validate heatmap availability
    """
    
    def __init__(
        self,
        match_repository: IMatchRepository,
        match_player_repository: IMatchPlayerRepository,
        summary_metrics_repository: ISummaryMetricsRepository,
        heatmap_repository: IHeatmapRepository
    ):
        self._match_repo = match_repository
        self._match_player_repo = match_player_repository
        self._metrics_repo = summary_metrics_repository
        self._heatmap_repo = heatmap_repository
    
    async def get_player_heatmap_raw(
        self, 
        match_id: int, 
        player_identifier: str
    ) -> bytes:
        """
        Get raw heatmap bytes for direct image response.
        Useful for <img src="/api/heatmaps/..."> usage.
        
        Returns:
            Raw PNG bytes
        """
        match = await self._match_repo.get_by_id(match_id)
        if not match:
            raise MatchNotFoundException(f"Match with ID {match_id} not found")
        
        match_player = await self._match_player_repo.get_by_identifier(
            match_id, player_identifier
        )
        if not match_player:
            raise PlayerInMatchNotFoundException(
                f"Player '{player_identifier}' not found in match {match_id}"
            )
        
        metrics = await self._metrics_repo.get_by_match_player_id(match_player.id)
        if not metrics or not metrics.heatmap_id:
            raise DataUnavailableException(
                f"Heatmap not available for player '{player_identifier}'"
            )
        
        heatmap = await self._heatmap_repo.get_by_id(metrics.heatmap_id)
        if not heatmap or not heatmap.heatmap:
            raise HeatmapNotFoundException(
                f"Heatmap data not found for player '{player_identifier}'"
            )
        
        return heatmap.heatmap