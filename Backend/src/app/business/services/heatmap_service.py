import base64
from typing import Optional, List
from dataclasses import dataclass

from .interfaces import IHeatmapService
from ..exceptions import (
    MatchNotFoundException,
    PlayerInMatchNotFoundException,
    DataUnavailableException,
    HeatmapNotFoundException,
    InsufficientPositionDataException
)
from ...data.repositories.interfaces import (
    IMatchRepository,
    IMatchPlayerRepository,
    ISummaryMetricsRepository,
    IHeatmapRepository
)


@dataclass
class HeatmapData:
    """Data transfer object for heatmap visualization"""
    player_identifier: str
    heatmap_base64: str
    content_type: str = "image/png"


@dataclass
class PlayerHeatmapResponse:
    """
    Response model for single player heatmap.
    Contains the 2D heatmap overlay data.
    """
    match_id: int
    player_identifier: str
    heatmap_2d: Optional[str]  # Base64 encoded PNG
    content_type: str = "image/png"


@dataclass
class HeatmapComparisonResponse:
    """
    Response model for comparing multiple player heatmaps.
    UC-02 S2: Comparison of heatmaps
    """
    match_id: int
    heatmaps: List[HeatmapData]


class HeatmapService(IHeatmapService):
    """
    Service for heatmap visualization business logic.
    
    Responsibilities:
    - Retrieve and encode heatmaps for API responses
    - Handle heatmap comparison logic
    - Validate heatmap availability
    
    Follows:
    - Interface Segregation Principle: Implements IHeatmapService
    - Single Responsibility Principle: Only handles heatmap logic
    - Dependency Inversion Principle: Depends on repository interfaces
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
    
    async def get_player_heatmap(
        self, 
        match_id: int, 
        player_identifier: str
    ) -> PlayerHeatmapResponse:
        """
        UC-02 S1: Get heatmap for a specific player in a match.
        
        Flow:
        1. Validate match exists
        2. Find MatchPlayer by identifier
        3. Get SummaryMetrics for that player
        4. Retrieve Heatmap via heatmap_id FK
        5. Encode as base64 for API response
        
        Args:
            match_id: ID of the match
            player_identifier: Player identifier (e.g., "player_1")
            
        Returns:
            PlayerHeatmapResponse with base64 encoded heatmap
            
        Raises:
            MatchNotFoundException: Match doesn't exist
            PlayerInMatchNotFoundException: Player not in match
            DataUnavailableException: Analysis not complete
            HeatmapNotFoundException: No heatmap data available
            InsufficientPositionDataException: Not enough tracking data (UC-02 F2)
        """
        # Step 1: Validate match exists
        match = await self._match_repo.get_by_id(match_id)
        if not match:
            raise MatchNotFoundException(f"Match with ID {match_id} not found")
        
        # Step 2: Find the player in the match
        match_player = await self._match_player_repo.get_by_identifier(
            match_id, player_identifier
        )
        if not match_player:
            raise PlayerInMatchNotFoundException(
                f"Player '{player_identifier}' not found in match {match_id}"
            )
        
        # Step 3: Get SummaryMetrics
        metrics = await self._metrics_repo.get_by_match_player_id(match_player.id)
        if not metrics:
            raise DataUnavailableException(
                f"Analysis data not available for player '{player_identifier}'"
            )
        
        # Step 4: Check if heatmap exists
        if not metrics.heatmap_id:
            # UC-02 F2: Insufficient position data
            raise InsufficientPositionDataException(
                f"Heatmap not available for player '{player_identifier}'. "
                "The AI could not track this player sufficiently."
            )
        
        # Step 5: Retrieve heatmap
        heatmap = await self._heatmap_repo.get_by_id(metrics.heatmap_id)
        if not heatmap or not heatmap.heatmap:
            raise HeatmapNotFoundException(
                f"Heatmap data not found for player '{player_identifier}'"
            )
        
        # Step 6: Encode as base64
        heatmap_base64 = base64.b64encode(heatmap.heatmap).decode('utf-8')
        
        return PlayerHeatmapResponse(
            match_id=match_id,
            player_identifier=player_identifier,
            heatmap_2d=heatmap_base64,
            content_type="image/png"
        )
    
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
    
    async def get_heatmap_comparison(
        self, 
        match_id: int,
        player_identifiers: Optional[List[str]] = None
    ) -> HeatmapComparisonResponse:
        """
        UC-02 S2: Get multiple heatmaps for comparison.
        
        Args:
            match_id: ID of the match
            player_identifiers: Optional list of players to compare.
                               If None, returns all available heatmaps.
                               
        Returns:
            HeatmapComparisonResponse with all requested heatmaps
        """
        match = await self._match_repo.get_by_id(match_id)
        if not match:
            raise MatchNotFoundException(f"Match with ID {match_id} not found")
        
        # Get all players in match
        match_players = await self._match_player_repo.get_by_match_id(match_id)
        
        # Filter if specific players requested
        if player_identifiers:
            match_players = [
                mp for mp in match_players 
                if mp.player_identifier in player_identifiers
            ]
        
        heatmaps: List[HeatmapData] = []
        
        for match_player in match_players:
            try:
                metrics = await self._metrics_repo.get_by_match_player_id(match_player.id)
                if not metrics or not metrics.heatmap_id:
                    continue  # Skip players without heatmap
                
                heatmap = await self._heatmap_repo.get_by_id(metrics.heatmap_id)
                if not heatmap or not heatmap.heatmap:
                    continue
                
                heatmap_base64 = base64.b64encode(heatmap.heatmap).decode('utf-8')
                heatmaps.append(HeatmapData(
                    player_identifier=match_player.player_identifier,
                    heatmap_base64=heatmap_base64
                ))
            except Exception:
                # Skip players with issues, continue with others
                continue
        
        if not heatmaps:
            raise DataUnavailableException(
                "No heatmap data available for any players in this match"
            )
        
        return HeatmapComparisonResponse(
            match_id=match_id,
            heatmaps=heatmaps
        )
    
    async def get_available_heatmap_players(self, match_id: int) -> List[str]:
        """
        Get list of players who have heatmap data available.
        Useful for UI to know which players can be displayed.
        
        Returns:
            List of player identifiers with available heatmaps
        """
        match = await self._match_repo.get_by_id(match_id)
        if not match:
            raise MatchNotFoundException(f"Match with ID {match_id} not found")
        
        match_players = await self._match_player_repo.get_by_match_id(match_id)
        available = []
        
        for mp in match_players:
            metrics = await self._metrics_repo.get_by_match_player_id(mp.id)
            if metrics and metrics.heatmap_id:
                heatmap = await self._heatmap_repo.get_by_id(metrics.heatmap_id)
                if heatmap and heatmap.heatmap:
                    available.append(mp.player_identifier)
        
        return available