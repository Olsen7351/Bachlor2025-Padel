"""
Dependency Injection Container

This module is the ONLY place where cross-layer wiring happens.
It knows about all layers and wires them together, but exposes
only abstractions (interfaces) to the presentation layer.

Architecture:
- Controllers import service factories from here
- Controllers NEVER import repositories directly
- All repository → service wiring is encapsulated here
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

# === Data Layer Imports ===
from .data.connection import get_db_session
from .data.repositories.player_repository import PlayerRepository
from .data.repositories.video_repository import VideoRepository
from .data.repositories.match_repository import MatchRepository, MatchPlayerRepository
from .data.repositories.analysis_repository import AnalysisRepository
from .data.repositories.summary_metrics_repository import SummaryMetricsRepository
from .data.repositories.rally_repository import RallyRepository
from .data.repositories.heatmap_repository import HeatmapRepository
from .data.repositories.hits_repository import HitsRepository

# === Business Layer Imports ===
from .business.services.player_service import PlayerService
from .business.services.video_service import VideoService
from .business.services.match_service import MatchService
from .business.services.rally_service import RallyService
from .business.services.heatmap_service import HeatmapService
from .business.services.analysis_service import AnalysisService
from .business.services.file_storage import FileStorageService
from .business.services.video_converter import VideoConverter
from .business.services.ml_service import MLService

# === Interface Exports (what controllers will use) ===
from .business.services.interfaces import (
    IPlayerService,
    IVideoService,
    IMatchService,
    IRallyService,
    IHeatmapService,
    IAnalysisService,
    IFileStorageService
)


# =============================================================================
# Service Factory Functions
# =============================================================================

async def get_player_service(
    session: AsyncSession = Depends(get_db_session)
) -> IPlayerService:
    """
    Factory for PlayerService.
    
    Wires:
    - PlayerRepository → PlayerService
    
    Used by:
    - PlayerController
    - AuthController
    """
    player_repository = PlayerRepository(session)
    return PlayerService(player_repository)


async def get_video_service(
    session: AsyncSession = Depends(get_db_session)
) -> IVideoService:
    """
    Factory for VideoService.
    
    Wires:
    - VideoRepository → VideoService
    - FileStorageService → VideoService
    - VideoConverter → VideoService
    
    Used by:
    - VideoController
    """
    video_repository = VideoRepository(session)
    file_storage_service = FileStorageService()
    video_converter = VideoConverter()
    
    return VideoService(
        video_repository=video_repository,
        file_storage_service=file_storage_service,
        video_converter=video_converter
    )


async def get_match_service(
    session: AsyncSession = Depends(get_db_session)
) -> IMatchService:
    """
    Factory for MatchService.
    
    Wires:
    - MatchRepository → MatchService
    - MatchPlayerRepository → MatchService
    - SummaryMetricsRepository → MatchService
    - AnalysisRepository → MatchService
    - HitsRepository → MatchService
    
    Used by:
    - MatchController
    """
    return MatchService(
        match_repository=MatchRepository(session),
        match_player_repository=MatchPlayerRepository(session),
        summary_metrics_repository=SummaryMetricsRepository(session),
        analysis_repository=AnalysisRepository(session),
        hits_repository=HitsRepository(session)
    )


async def get_rally_service(
    session: AsyncSession = Depends(get_db_session)
) -> IRallyService:
    """
    Factory for RallyService.
    
    Wires:
    - MatchRepository → RallyService
    - SummaryMetricsRepository → RallyService
    - RallyRepository → RallyService
    
    Used by:
    - RallyController
    """
    return RallyService(
        match_repository=MatchRepository(session),
        summary_metrics_repository=SummaryMetricsRepository(session),
        rally_repository=RallyRepository(session)
    )


async def get_heatmap_service(
    session: AsyncSession = Depends(get_db_session)
) -> IHeatmapService:
    """
    Factory for HeatmapService.
    
    Wires:
    - MatchRepository → HeatmapService
    - MatchPlayerRepository → HeatmapService
    - SummaryMetricsRepository → HeatmapService
    - HeatmapRepository → HeatmapService
    
    Used by:
    - HeatmapController
    """
    return HeatmapService(
        match_repository=MatchRepository(session),
        match_player_repository=MatchPlayerRepository(session),
        summary_metrics_repository=SummaryMetricsRepository(session),
        heatmap_repository=HeatmapRepository(session)
    )


async def get_analysis_service(
    session: AsyncSession = Depends(get_db_session)
) -> IAnalysisService:
    """
    Factory for AnalysisService.
    
    Wires:
    - All analysis-related repositories → AnalysisService
    - MLService → AnalysisService
    
    Used by:
    - VideoController (background task)
    """
    return AnalysisService(
        analysis_repository=AnalysisRepository(session),
        match_repository=MatchRepository(session),
        match_player_repository=MatchPlayerRepository(session),
        summary_metrics_repository=SummaryMetricsRepository(session),
        hits_repository=HitsRepository(session),
        rally_repository=RallyRepository(session),
        heatmap_repository=HeatmapRepository(session),
        video_repository=VideoRepository(session),
        ml_service=MLService()
    )


# =============================================================================
# Standalone Service Factories (no database session needed)
# =============================================================================

def get_file_storage_service() -> IFileStorageService:
    """
    Factory for FileStorageService.
    
    No database dependencies - just file system operations.
    """
    return FileStorageService()


# =============================================================================
# Factory Functions for Background Tasks
# =============================================================================

def create_analysis_service(session: AsyncSession) -> AnalysisService:
    """
    Create AnalysisService for use in background tasks.
    
    Unlike the async factory above, this accepts session directly
    since background tasks manage their own session lifecycle.
    """
    return AnalysisService(
        analysis_repository=AnalysisRepository(session),
        match_repository=MatchRepository(session),
        match_player_repository=MatchPlayerRepository(session),
        summary_metrics_repository=SummaryMetricsRepository(session),
        hits_repository=HitsRepository(session),
        rally_repository=RallyRepository(session),
        heatmap_repository=HeatmapRepository(session),
        video_repository=VideoRepository(session),
        ml_service=MLService()
    )