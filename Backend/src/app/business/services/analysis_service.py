from typing import Optional
from datetime import datetime
from pathlib import Path

from .interfaces import IAnalysisService, IMLService
from ...domain.analysis import Analysis
from ...domain.match import Match, MatchPlayer, SummaryMetrics, Hits, Rally, Heatmap
from ...domain.video import VideoStatus
from ..exceptions import (
    VideoNotFoundException,
    AnalysisException
)
from ...data.repositories.interfaces import (
    IAnalysisRepository,
    IMatchRepository,
    IMatchPlayerRepository,
    ISummaryMetricsRepository,
    IHitsRepository,
    IRallyRepository,
    IHeatmapRepository,
    IVideoRepository
)
from .deep_learning.results import MLAnalysisResult, TRACKED_PLAYERS

class AnalysisService(IAnalysisService):
    """
    Analysis service - Orchestrates the creation and execution of video analysis.
    
    This service:
    1. Creates the entity chain when analysis starts
    2. Runs ML inference via MLService
    3. Stores results in the database
    4. Updates video status throughout
    
    Note: Only player_1 and player_2 are tracked (near-side of court).
    """
    
    def __init__(
        self,
        analysis_repository: IAnalysisRepository,
        match_repository: IMatchRepository,
        match_player_repository: IMatchPlayerRepository,
        summary_metrics_repository: ISummaryMetricsRepository,
        hits_repository: IHitsRepository,
        rally_repository: IRallyRepository,
        heatmap_repository: IHeatmapRepository,
        video_repository: IVideoRepository,
        ml_service: IMLService
    ):
        """Initialize with all required repositories and services"""
        self._analysis_repo = analysis_repository
        self._match_repo = match_repository
        self._match_player_repo = match_player_repository
        self._metrics_repo = summary_metrics_repository
        self._hits_repo = hits_repository
        self._rally_repo = rally_repository
        self._heatmap_repo = heatmap_repository
        self._video_repo = video_repository
        self._ml_service = ml_service
    
    async def create_analysis_for_video(
        self, 
        video_id: int, 
        player_id: str
    ) -> Analysis:
        """
        Create analysis entity chain for a video.
        Called AFTER video upload completes.
        
        Creates:
        1. Analysis entity
        2. Match entity
        3. 2 MatchPlayer entities (player_1 and player_2 ONLY)
        4. Updates Video status to PROCESSING
        
        Note: Only near-side players are created. Far-side players 
        (player_3, player_4) don't get meaningful tracking data from ML.
        
        Note: SummaryMetrics, Hits, Heatmap created later after ML analysis.
        
        Args:
            video_id: ID of the uploaded video
            player_id: Firebase UID of the player who uploaded
            
        Returns:
            Created Analysis entity
        """
        video = await self._video_repo.get_by_id(video_id)
        if not video:
            raise VideoNotFoundException(f"Video with ID {video_id} not found")
        
        try:
            # Step 1: Create Analysis entity
            analysis = Analysis(
                id=None,
                player_id=player_id,
                video_id=video_id,
                match_id=None,
                analysis_timestamp=datetime.now(),
                created_at=None,
                updated_at=None
            )
            created_analysis = await self._analysis_repo.create(analysis)
            
            # Step 2: Create Match entity
            match = Match(id=None, created_at=None, updated_at=None)
            created_match = await self._match_repo.create(match)
            
            # Step 3: Update Analysis with match_id
            created_analysis.match_id = created_match.id
            updated_analysis = await self._analysis_repo.update(created_analysis)
            
            # Step 4: Create MatchPlayer entities - only for tracked players
            # Uses TRACKED_PLAYERS constant from ml_service: ["player_1", "player_2"]
            for player_identifier in TRACKED_PLAYERS:
                match_player = MatchPlayer(
                    id=None,
                    match_id=created_match.id,
                    player_identifier=player_identifier,
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
            await self._video_repo.update_status(
                video_id=video_id,
                status=VideoStatus.ERROR
            )
            raise AnalysisException(f"Failed to create analysis: {str(e)}")
    
    async def run_ml_analysis(
        self,
        video_id: int,
        video_path: Path,
        court_number: int,
        fps: float = 30.0
    ) -> MLAnalysisResult:
        """
        Run ML inference on a video.
        
        Args:
            video_id: ID of the video being analyzed
            video_path: Path to the video file
            court_number: Court number for calibration
            fps: Video framerate
            
        Returns:
            MLAnalysisResult with all analysis data
        """
        try:
            result = await self._ml_service.run_analysis(
                video_path=video_path,
                court_number=court_number,
                fps=fps
            )
            return result
        except Exception as e:
            await self._video_repo.update_status(
                video_id=video_id,
                status=VideoStatus.ERROR
            )
            raise AnalysisException(f"ML analysis failed: {str(e)}")
    
    async def store_analysis_results(
        self,
        match_id: int,
        ml_result: MLAnalysisResult
    ) -> None:
        """
        Store ML analysis results in the database.
        
        Flow for each player (player_1 and player_2 only):
        1. Create Hits record (standalone)
        2. Create Heatmap record (standalone)
        3. Create SummaryMetrics with hits_id and heatmap_id FKs
        4. Create Rally records with summary_metrics_id (only for player_1)
        
        Note: Since we now only create 2 MatchPlayer entities, all players
        in match_players will have detailed data.
        
        Args:
            match_id: ID of the match
            ml_result: Results from ML pipeline
        """
        try:
            match_players = await self._match_player_repo.get_by_match_id(match_id)
            
            rallies_stored = False
            
            for match_player in match_players:
                player_id = match_player.player_identifier
                player_stats = ml_result.player_stats.get(player_id)
                
                hits_id = None
                heatmap_id = None
                
                # All players now get detailed data (since we only create player_1, player_2)
                # Step 1: Create Hits record (standalone)
                if player_stats:
                    hits = Hits(
                        id=None,
                        overhead_hits=player_stats.overhead_hits,
                        lob=player_stats.lob,
                        serve=player_stats.serve,
                        groundstrokes=player_stats.groundstrokes,
                        created_at=None,
                        updated_at=None
                    )
                    created_hits = await self._hits_repo.create(hits)
                    hits_id = created_hits.id
                
                # Step 2: Create Heatmap record (standalone)
                heatmap_data = ml_result.heatmaps.get(player_id)
                if heatmap_data and heatmap_data.image_path.exists():
                    image_bytes = self._ml_service.read_heatmap_binary(
                        heatmap_data.image_path
                    )
                    if image_bytes:
                        heatmap = Heatmap(
                            id=None,
                            heatmap=image_bytes,
                            created_at=None,
                            updated_at=None
                        )
                        created_heatmap = await self._heatmap_repo.create(heatmap)
                        heatmap_id = created_heatmap.id
                
                # Step 3: Create SummaryMetrics with FKs
                total_hits = player_stats.total_hits if player_stats else 0
                metrics = SummaryMetrics(
                    id=None,
                    match_player_id=match_player.id,
                    total_hits=total_hits,
                    total_rallies=ml_result.total_rallies,
                    hits_id=hits_id,
                    heatmap_id=heatmap_id,
                    created_at=None,
                    updated_at=None
                )
                created_metrics = await self._metrics_repo.create(metrics)
                
                # Step 4: Store rallies once (match-level, not player-specific)
                # Store under first player to avoid duplication
                if not rallies_stored:
                    for rally_data in ml_result.rallies:
                        rally = Rally(
                            id=None,
                            summary_metrics_id=created_metrics.id,
                            duration=rally_data.duration_seconds,
                            created_at=None,
                            updated_at=None
                        )
                        await self._rally_repo.create(rally)
                    rallies_stored = True
                        
        except Exception as e:
            raise AnalysisException(f"Failed to store analysis results: {str(e)}")
    
    async def complete_analysis(
        self,
        video_id: int,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Mark analysis as complete (or failed)"""
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
    