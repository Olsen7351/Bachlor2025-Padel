from typing import Dict

from .interfaces import IMatchService
from ..exceptions import (
    MatchNotFoundException,
    DataUnavailableException
)
from ...data.repositories.interfaces import (
    IMatchRepository,
    IMatchPlayerRepository,
    ISummaryMetricsRepository,
    IAnalysisRepository,
    IHitsRepository
)


class MatchService(IMatchService):
    """
    Match service implementation
    Handles business logic for UC-04: Display total hit counts
    
    Responsibilities:
    - Orchestrate match-related business operations
    - Implement UC-04 success and failure scenarios
    - Validate business rules
    - Coordinate between repositories
    """
    
    def __init__(
        self,
        match_repository: IMatchRepository,
        match_player_repository: IMatchPlayerRepository,
        summary_metrics_repository: ISummaryMetricsRepository,
        analysis_repository: IAnalysisRepository,
        hits_repository: IHitsRepository
    ):
        """
        Initialize service with dependencies (Dependency Injection)
        
        Args:
            match_repository: Repository for match data access
            match_player_repository: Repository for match player data access
            summary_metrics_repository: Repository for summary metrics data access
            analysis_repository: Repository for analysis data access
        """
        self._match_repo = match_repository
        self._match_player_repo = match_player_repository
        self._metrics_repo = summary_metrics_repository
        self._analysis_repo = analysis_repository
        self._hits_repo = hits_repository
    
    async def get_match_overview(self, match_id: int) -> Dict:
        """
        Get match overview with player statistics
        Implements UC-04 Success Scenario S1
        
        Business Rules:
        1. Match must exist
        2. Hit data must be available for all players
        3. Returns list of players with their total hit counts
        4. Analysis ID is retrieved via the 1:1 relationship with Match
        
        Returns:
            Dictionary containing:
            - match_id: int
            - analysis_id: int (retrieved from Analysis entity)
            - player_statistics: List of player hit counts
            - created_at: datetime
            
        Raises:
            MatchNotFoundException: If match doesn't exist
            DataUnavailableException: If hit data is not available (UC-04 F1)
        """
        # Verify match exists
        match = await self._match_repo.get_by_id(match_id)
        if not match:
            raise MatchNotFoundException(f"Match with ID {match_id} not found")
        
        # Get analysis for this match (1:1 relationship)
        # Analysis has match_id FK, so we query by match_id
        analysis = await self._analysis_repo.get_by_match_id(match_id)
        analysis_id = analysis.id if analysis else None
        
        # Get all metrics for the match
        metrics_list = await self._metrics_repo.get_all_by_match_id(match_id)
        
        # UC-04 F1: Check if data is available
        if not metrics_list:
            raise DataUnavailableException(
                "Hit data is not available for this match. "
                "The analysis may not have completed successfully."
            )
        
        # Get player information for each metric
        player_statistics = []
        for metrics in metrics_list:
            # Get the MatchPlayer to get player_identifier
            match_player = await self._match_player_repo.get_by_id(metrics.match_player_id)

            hits_details = None
            if metrics.hits_id:
                hits_details = await self._hits_repo.get_by_id(metrics.hits_id)

            if match_player:
                player_statistics.append({
                    "player_identifier": match_player.player_identifier,
                    "total_hits": metrics.total_hits,
                    "overhead_hits": hits_details.overhead_hits if hits_details else 0,
                    "lob": hits_details.lob if hits_details else 0,
                    "serve": hits_details.serve if hits_details else 0,
                    "groundstrokes": hits_details.groundstrokes if hits_details else 0
                })
        
        return {
            "match_id": match.id,
            "analysis_id": analysis_id,
            "player_statistics": player_statistics,
            "created_at": match.created_at
        }