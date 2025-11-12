from typing import List, Optional, Dict
from datetime import datetime

from ...domain.match import Match, MatchPlayer, SummaryMetrics
from .interfaces import IMatchService
from ..exceptions import (
    MatchNotFoundException,
    PlayerInMatchNotFoundException,
    DataUnavailableException,
    InvalidSetNumberException
)
from ...data.repositories.interfaces import (
    IMatchRepository,
    IMatchPlayerRepository,
    ISummaryMetricsRepository,
    IAnalysisRepository
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
    
    Follows:
    - Single Responsibility Principle: Only handles match business logic
    - Dependency Inversion Principle: Depends on repository interfaces
    - Open/Closed Principle: Extensible through inheritance
    """
    
    def __init__(
        self,
        match_repository: IMatchRepository,
        match_player_repository: IMatchPlayerRepository,
        summary_metrics_repository: ISummaryMetricsRepository,
        analysis_repository: IAnalysisRepository
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
            if match_player:
                player_statistics.append({
                    "player_identifier": match_player.player_identifier,
                    "total_hits": metrics.total_hits
                })
        
        return {
            "match_id": match.id,
            "analysis_id": analysis_id,
            "player_statistics": player_statistics,
            "created_at": match.created_at
        }
    
    async def get_match_statistics_by_set(self, match_id: int, set_number: int) -> Dict:
        """
        Get match statistics filtered by set
        Implements UC-04 Success Scenario S2
        
        Business Rules:
        1. Match must exist
        2. Set number must be valid (>= 1)
        3. Data must be available for the specified set
        
        Note: Set-level filtering requires Set entity implementation.
        Currently returns match-level statistics with set_number in response.
        
        Returns:
            Dictionary containing set-specific statistics:
            - match_id: int
            - set_number: int
            - player_statistics: List of player hit counts
            
        Raises:
            MatchNotFoundException: If match doesn't exist
            InvalidSetNumberException: If set number is invalid
            DataUnavailableException: If hit data is not available
        """
        # Validate set number
        if set_number < 1:
            raise InvalidSetNumberException(f"Set number must be >= 1, got {set_number}")
        
        # Verify match exists
        match = await self._match_repo.get_by_id(match_id)
        if not match:
            raise MatchNotFoundException(f"Match with ID {match_id} not found")
        
        # Get analysis for this match
        analysis = await self._analysis_repo.get_by_match_id(match_id)
        analysis_id = analysis.id if analysis else None
        
        # Get metrics filtered by set
        # NOTE: Currently returns all match metrics until set-level tracking is implemented
        metrics_list = await self._metrics_repo.get_by_match_and_set(match_id, set_number)
        
        # UC-04 F1: Check if data is available
        if not metrics_list:
            raise DataUnavailableException(
                f"Hit data is not available for set {set_number}. "
                "The analysis may not have completed successfully."
            )
        
        # Build response with player statistics
        player_statistics = []
        for metrics in metrics_list:
            match_player = await self._match_player_repo.get_by_id(metrics.match_player_id)
            if match_player:
                player_statistics.append({
                    "player_identifier": match_player.player_identifier,
                    "total_hits": metrics.total_hits
                })
        
        return {
            "match_id": match.id,
            "analysis_id": analysis_id,
            "set_number": set_number,
            "player_statistics": player_statistics
        }
    
    async def get_hit_comparison_chart_data(self, match_id: int) -> Dict:
        """
        Get data formatted for visual hit comparison chart
        Implements UC-04 Success Scenario S3
        
        Business Rules:
        1. Match must exist
        2. Data must be available for all players
        3. Data formatted for bar chart visualization
        4. Players sorted by hit count (descending) for better visualization
        
        Returns:
            Dictionary formatted for chart visualization:
            - chart_type: "bar"
            - players: List of players with hit counts (sorted)
            
        Raises:
            MatchNotFoundException: If match doesn't exist
            DataUnavailableException: If hit data is not available
        """
        # Verify match exists
        match = await self._match_repo.get_by_id(match_id)
        if not match:
            raise MatchNotFoundException(f"Match with ID {match_id} not found")
        
        # Get all metrics for the match
        metrics_list = await self._metrics_repo.get_all_by_match_id(match_id)
        
        # UC-04 F1: Check if data is available
        if not metrics_list:
            raise DataUnavailableException(
                "Hit data is not available for this match. "
                "Cannot generate comparison chart."
            )
        
        # Build chart data
        players = []
        for metrics in metrics_list:
            match_player = await self._match_player_repo.get_by_id(metrics.match_player_id)
            if match_player:
                players.append({
                    "player_identifier": match_player.player_identifier,
                    "total_hits": metrics.total_hits
                })
        
        # Sort by hit count descending for better visualization
        # Note: Repository already sorts, but we do it here for clarity
        players.sort(key=lambda x: x["total_hits"], reverse=True)
        
        return {
            "chart_type": "bar",
            "players": players
        }
    
    async def get_player_hit_count(self, match_id: int, player_identifier: str) -> int:
        """
        Get hit count for a specific player in a match
        
        Business Rules:
        1. Match must exist
        2. Player must exist in the match
        3. Hit data must be available
        
        Returns:
            Total hit count for the player
            
        Raises:
            MatchNotFoundException: If match doesn't exist
            PlayerInMatchNotFoundException: If player not found in match
            DataUnavailableException: If hit data is not available
        """
        # Verify match exists
        match = await self._match_repo.get_by_id(match_id)
        if not match:
            raise MatchNotFoundException(f"Match with ID {match_id} not found")
        
        # Find player in match
        match_player = await self._match_player_repo.get_by_identifier(match_id, player_identifier)
        if not match_player:
            raise PlayerInMatchNotFoundException(
                f"Player '{player_identifier}' not found in match {match_id}"
            )
        
        # Get metrics for this player
        metrics = await self._metrics_repo.get_by_match_player_id(match_player.id)
        
        # UC-04 F1: Check if data is available
        if not metrics:
            raise DataUnavailableException(
                f"Hit data is not available for player '{player_identifier}'. "
                "The analysis may not have completed successfully."
            )
        
        return metrics.total_hits