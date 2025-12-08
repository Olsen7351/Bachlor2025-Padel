from typing import Dict, Any
from datetime import datetime

from ...domain.analysis import Analysis
from ...domain.match import Match, MatchPlayer, SummaryMetrics
from ...domain.video import Video, VideoStatus
from ..exceptions import (
    AnalysisNotFoundException,
    VideoNotFoundException,
    AnalysisException
)
from ...data.repositories.interfaces import (
    IAnalysisRepository,
    IMatchRepository,
    IMatchPlayerRepository,
    ISummaryMetricsRepository,
    IVideoRepository
)


class AnalysisService:
    """
    Analysis service - Orchestrates the creation of analysis entities
    This is the key service that creates the entity chain for UC-04
    """
    
    def __init__(
        self,
        analysis_repository: IAnalysisRepository,
        match_repository: IMatchRepository,
        match_player_repository: IMatchPlayerRepository,
        summary_metrics_repository: ISummaryMetricsRepository,
        video_repository: IVideoRepository
    ):
        """Initialize with all required repositories"""
        self._analysis_repo = analysis_repository
        self._match_repo = match_repository
        self._match_player_repo = match_player_repository
        self._metrics_repo = summary_metrics_repository
        self._video_repo = video_repository
    
    async def create_analysis_for_video(
        self, 
        video_id: int, 
        player_id: str
    ) -> Analysis:
        """
        Create analysis entity chain for a video
        This is called AFTER video upload completes (UC-01)
        
        Creates:
        1. Analysis entity
        2. Match entity
        3. 4 MatchPlayer entities (player_1 through player_4)
        4. Updates Analysis with match_id
        5. Updates Video status to PROCESSING
        
        Args:
            video_id: ID of the uploaded video
            player_id: Firebase UID of the player who uploaded
            
        Returns:
            Created Analysis entity
            
        Raises:
            VideoNotFoundException: If video doesn't exist
            AnalysisException: If creation fails
        """
        # Verify video exists
        video = await self._video_repo.get_by_id(video_id)
        if not video:
            raise VideoNotFoundException(f"Video with ID {video_id} not found")
        
        try:
            # Step 1: Create Analysis entity (without match_id initially)
            analysis = Analysis(
                id=None,
                player_id=player_id,
                video_id=video_id,
                match_id=None,  # Will be set after Match creation
                analysis_timestamp=datetime.now(),
                created_at=None,  # Set by repository
                updated_at=None   # Set by repository
            )
            created_analysis = await self._analysis_repo.create(analysis)
            
            # Step 2: Create Match entity
            match = Match(
                id=None,
                created_at=None,
                updated_at=None
            )
            created_match = await self._match_repo.create(match)
            
            # Step 3: Update Analysis with match_id
            created_analysis.match_id = created_match.id
            updated_analysis = await self._analysis_repo.update(created_analysis)
            
            # Step 4: Create 4 MatchPlayer entities (as detected by ML model)
            for i in range(1, 5):
                match_player = MatchPlayer(
                    id=None,
                    match_id=created_match.id,
                    player_identifier=f"player_{i}",
                    created_at=None,
                    updated_at=None
                )
                await self._match_player_repo.create(match_player)
            
            # Step 5: Update Video status to PROCESSING
            await self._video_repo.update_status(
                video_id=video_id,
                status=VideoStatus.PROCESSING
            )
            
            return updated_analysis
            
        except Exception as e:
            # If anything fails, set video to ERROR status
            await self._video_repo.update_status(
                video_id=video_id,
                status=VideoStatus.ERROR
            )
            raise AnalysisException(f"Failed to create analysis: {str(e)}")
    
    async def store_analysis_results(
        self,
        match_id: int,
        ai_results: Dict[str, Any]
    ) -> None:
        """
        Store AI analysis results as SummaryMetrics
        Called AFTER AI model completes analysis
        
        Args:
            match_id: ID of the match
            ai_results: Dictionary with player stats from AI model
                Format: {
                    "player_1": {"hits": 245, "rallies": 45},
                    "player_2": {"hits": 238, "rallies": 45},
                    "player_3": {"hits": 221, "rallies": 45},
                    "player_4": {"hits": 215, "rallies": 45}
                }
                
        Raises:
            AnalysisException: If storing results fails
        """
        try:
            # Get all match players
            match_players = await self._match_player_repo.get_by_match_id(match_id)
            
            # Create SummaryMetrics for each player
            for match_player in match_players:
                player_stats = ai_results.get(match_player.player_identifier, {})
                
                metrics = SummaryMetrics(
                    id=None,
                    match_player_id=match_player.id,
                    total_hits=player_stats.get("hits", 0),
                    total_rallies=player_stats.get("rallies", 0),
                    created_at=None,
                    updated_at=None
                )
                await self._metrics_repo.create(metrics)
                
        except Exception as e:
            raise AnalysisException(f"Failed to store analysis results: {str(e)}")
    
    async def complete_analysis(
        self,
        video_id: int,
        success: bool = True,
        error_message: str = None
    ) -> None:
        """
        Mark analysis as complete (or failed)
        Updates video status to ANALYZED or ERROR
        
        Args:
            video_id: ID of the video
            success: Whether analysis succeeded
            error_message: Optional error message if failed
        """
        if success:
            await self._video_repo.update_status(
                video_id=video_id,
                status=VideoStatus.ANALYZED
            )
        else:
            await self._video_repo.update_status(
                video_id=video_id,
                status=VideoStatus.ERROR
            )
    
    async def get_analysis_by_video(self, video_id: int) -> Analysis:
        """Get analysis for a video"""
        analysis = await self._analysis_repo.get_by_video_id(video_id)
        if not analysis:
            raise AnalysisNotFoundException(f"No analysis found for video {video_id}")
        return analysis