from typing import Dict

from .interfaces import IRallyService
from ..exceptions import (
    MatchNotFoundException,
    DataUnavailableException,
    RallyDataUnavailableException
)
from ...data.repositories.interfaces import (
    IMatchRepository,
    ISummaryMetricsRepository,
    IRallyRepository
)

class RallyService(IRallyService):
    """
    Rally Service Implementation
    
    Handles UC-08: Rally Length Analysis
    
    Responsibilities:
    - Retrieve and analyze rally data for a match
    - Calculate statistics (average, min, max duration)
    - Handle failure scenarios when data is unavailable
    """
    
    def __init__(
        self,
        match_repository: IMatchRepository,
        summary_metrics_repository: ISummaryMetricsRepository,
        rally_repository: IRallyRepository
    ):
        """
        Initialize service with dependencies (Dependency Injection)
        
        Args:
            match_repository: Repository for match data access
            summary_metrics_repository: Repository for summary metrics
            rally_repository: Repository for rally data access
        """
        self._match_repo = match_repository
        self._metrics_repo = summary_metrics_repository
        self._rally_repo = rally_repository
    
    async def get_rally_analysis(self, match_id: int) -> Dict:
        """
        UC-08 S1: Get rally analysis for a match
        
        Returns overview of rally lengths including:
        - Total number of rallies
        - Average rally duration
        - Min/Max duration
        - Distribution of rally lengths
        
        Business Rules:
        1. Match must exist
        2. Rally data must be available
        3. At least one rally required for meaningful analysis
        
        Returns:
            Dictionary containing:
            - match_id: int
            - total_rallies: int
            - average_duration: float (seconds)
            - min_duration: float (seconds)
            - max_duration: float (seconds)
            - rallies: List of rally details
            
        Raises:
            MatchNotFoundException: If match doesn't exist
            RallyDataUnavailableException: UC-08 F1 - If rally data not available
        """
        # Verify match exists
        match = await self._match_repo.get_by_id(match_id)
        if not match:
            raise MatchNotFoundException(f"Match with ID {match_id} not found")
        
        # Get summary metrics to find rally data
        metrics_list = await self._metrics_repo.get_all_by_match_id(match_id)
        
        if not metrics_list:
            raise DataUnavailableException(
                "No analysis data available for this match. "
                "The analysis may not have completed successfully."
            )
        
        # Get rallies from the first summary metrics (rallies are match-level)
        # They're stored via SummaryMetrics but represent the whole match
        all_rallies = []
        for metrics in metrics_list:
            rallies = await self._rally_repo.get_by_summary_metrics_id(metrics.id)
            if rallies:
                all_rallies = rallies
                break  # Only need one set - rallies are shared
        
        # UC-08 F1: Check if rally data is available
        if not all_rallies:
            raise RallyDataUnavailableException(
                "Rally data is not available for this match. "
                "The AI could not distinguish individual rallies from each other."
            )
        
        # Calculate statistics
        durations = [rally.duration for rally in all_rallies]
        total = len(durations)
        avg_duration = sum(durations) / total if total > 0 else 0.0
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0
        
        # Build rally details list
        rally_details = [
            {
                "rally_id": rally.id,
                "duration": rally.duration
            }
            for rally in all_rallies
        ]
        
        return {
            "match_id": match.id,
            "total_rallies": total,
            "average_duration": round(avg_duration, 2),
            "min_duration": round(min_duration, 2),
            "max_duration": round(max_duration, 2),
            "rallies": rally_details
        }
    
    async def get_rally_duration_distribution(self, match_id: int) -> Dict:
        """
        Get rally duration distribution for visualization
        
        Groups rallies into duration buckets for histogram/chart display.
        
        Buckets:
        - Short: < 5 seconds
        - Medium: 5-15 seconds
        - Long: 15-30 seconds
        - Very Long: > 30 seconds
        
        Returns:
            Dictionary with bucket counts and percentages
        """
        # Get base analysis first
        analysis = await self.get_rally_analysis(match_id)
        
        # Define buckets
        buckets = {
            "short": {"label": "< 5s", "min": 0, "max": 5, "count": 0},
            "medium": {"label": "5-15s", "min": 5, "max": 15, "count": 0},
            "long": {"label": "15-30s", "min": 15, "max": 30, "count": 0},
            "very_long": {"label": "> 30s", "min": 30, "max": float('inf'), "count": 0}
        }
        
        # Categorize rallies
        for rally in analysis["rallies"]:
            duration = rally["duration"]
            for bucket_key, bucket in buckets.items():
                if bucket["min"] <= duration < bucket["max"]:
                    bucket["count"] += 1
                    break
        
        # Calculate percentages
        total = analysis["total_rallies"]
        distribution = []
        for bucket_key, bucket in buckets.items():
            count = bucket["count"]
            percentage = (count / total * 100) if total > 0 else 0
            distribution.append({
                "bucket": bucket_key,
                "label": bucket["label"],
                "count": count,
                "percentage": round(percentage, 1)
            })
        
        return {
            "match_id": match_id,
            "total_rallies": total,
            "distribution": distribution
        }