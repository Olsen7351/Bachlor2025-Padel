from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, Optional

from app.presentation.dtos.match_dto import (
    MatchSummaryDto,
    HitComparisonChartDto,
    PlayerHitCountDto,
    DataUnavailableResponse,
    MatchErrorResponse
)
from ...business.services.match_service import MatchService
from ...data.repositories.match_repository import MatchRepository, MatchPlayerRepository
from ...data.repositories.summary_metrics_repository import SummaryMetricsRepository
from ...data.repositories.analysis_repository import AnalysisRepository
from ...data.connection import get_db_session
from ...auth.dependencies import get_current_user
from ...domain.player import Player
from ...business.exceptions import (
    MatchNotFoundException,
    PlayerInMatchNotFoundException,
    DataUnavailableException,
    InvalidSetNumberException
)


router = APIRouter(prefix="/matches", tags=["matches"])


def get_match_service(session: AsyncSession = Depends(get_db_session)) -> MatchService:
    """
    Dependency injection for MatchService
    
    Follows Dependency Inversion Principle:
    - Controller depends on abstractions (service interfaces)
    - Service depends on abstractions (repository interfaces)
    - Concrete implementations injected at runtime
    """
    match_repository = MatchRepository(session)
    match_player_repository = MatchPlayerRepository(session)
    summary_metrics_repository = SummaryMetricsRepository(session)
    analysis_repository = AnalysisRepository(session)
    
    return MatchService(
        match_repository,
        match_player_repository,
        summary_metrics_repository,
        analysis_repository  # Now includes analysis repository
    )


@router.get(
    "/{match_id}/overview",
    response_model=MatchSummaryDto,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": MatchErrorResponse, "description": "Match not found"},
        503: {"model": DataUnavailableResponse, "description": "Hit data not available"}
    },
    summary="Get match overview with player statistics",
    description="""
    Get match overview displaying total hit counts for each player.
    Implements UC-04 Success Scenario S1.
    
    **Success Scenario S1:**
    - Display list of players with their total hit counts
    - Data sorted by hit count (highest first)
    - Includes analysis_id from the associated analysis
    
    **Failure Scenario F1:**
    - Returns 503 if hit data is not available
    - Analysis may have failed or not completed
    """
)
async def get_match_overview(
    match_id: int,
    current_user: Player = Depends(get_current_user),
    match_service: MatchService = Depends(get_match_service)
) -> MatchSummaryDto:
    """
    UC-04 S1: Get match overview with hit counts
    
    Returns match information with player statistics including total hits.
    Includes analysis_id retrieved from the 1:1 Analysis-Match relationship.
    """
    try:
        overview = await match_service.get_match_overview(match_id)
        
        # Map to DTO
        player_statistics = [
            PlayerHitCountDto(
                player_identifier=stat["player_identifier"],
                total_hits=stat["total_hits"]
            )
            for stat in overview["player_statistics"]
        ]
        
        return MatchSummaryDto(
            match_id=overview["match_id"],
            analysis_id=overview["analysis_id"],  # Now properly retrieved
            player_statistics=player_statistics,
            created_at=overview["created_at"]
        )
        
    except MatchNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": str(e)}
        )
    
    except DataUnavailableException as e:
        # UC-04 F1: Data not available
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "data_unavailable",
                "message": str(e),
                "reason": "Hit identification may have failed during analysis"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "An unexpected error occurred", "details": str(e)}
        )


@router.get(
    "/{match_id}/statistics",
    response_model=MatchSummaryDto,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": MatchErrorResponse, "description": "Match not found"},
        400: {"model": MatchErrorResponse, "description": "Invalid set number"},
        503: {"model": DataUnavailableResponse, "description": "Hit data not available"}
    },
    summary="Get match statistics with optional set filter",
    description="""
    Get match statistics with optional filtering by set number.
    Implements UC-04 Success Scenario S2.
    
    **Success Scenario S2:**
    - Filter statistics by specific set number
    - Shows only hit counts for the selected set
    
    **Note:** Set-level filtering is planned for future implementation.
    Currently returns match-level statistics.
    """
)
async def get_match_statistics_by_set(
    match_id: int,
    set_number: Annotated[Optional[int], Query(description="Filter by set number")] = None,
    current_user: Player = Depends(get_current_user),
    match_service: MatchService = Depends(get_match_service)
) -> MatchSummaryDto:
    """
    UC-04 S2: Get match statistics filtered by set
    
    Query parameters:
    - set_number: Optional set number to filter (1, 2, 3, etc.)
    """
    try:
        # If set_number provided, use set-filtered endpoint
        if set_number is not None:
            stats = await match_service.get_match_statistics_by_set(match_id, set_number)
        else:
            stats = await match_service.get_match_overview(match_id)
        
        # Map to DTO
        player_statistics = [
            PlayerHitCountDto(
                player_identifier=stat["player_identifier"],
                total_hits=stat["total_hits"]
            )
            for stat in stats["player_statistics"]
        ]
        
        return MatchSummaryDto(
            match_id=stats["match_id"],
            analysis_id=stats.get("analysis_id"),
            player_statistics=player_statistics,
            created_at=stats.get("created_at")
        )
        
    except MatchNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": str(e)}
        )
    
    except InvalidSetNumberException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)}
        )
    
    except DataUnavailableException as e:
        # UC-04 F1: Data not available
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "data_unavailable",
                "message": str(e),
                "reason": "Hit identification may have failed during analysis"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "An unexpected error occurred", "details": str(e)}
        )


@router.get(
    "/{match_id}/hit-comparison",
    response_model=HitComparisonChartDto,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": MatchErrorResponse, "description": "Match not found"},
        503: {"model": DataUnavailableResponse, "description": "Hit data not available"}
    },
    summary="Get hit comparison chart data",
    description="""
    Get data formatted for visual hit comparison chart.
    Implements UC-04 Success Scenario S3.
    
    **Success Scenario S3:**
    - Returns data formatted for bar chart visualization
    - Compares hit counts across all players
    - Players sorted by hit count (highest first)
    """
)
async def get_hit_comparison_chart(
    match_id: int,
    current_user: Player = Depends(get_current_user),
    match_service: MatchService = Depends(get_match_service)
) -> HitComparisonChartDto:
    """
    UC-04 S3: Get visual comparison of player hit counts
    
    Returns data formatted for chart visualization (bar chart)
    """
    try:
        chart_data = await match_service.get_hit_comparison_chart_data(match_id)
        
        # Map to DTO
        players = [
            PlayerHitCountDto(
                player_identifier=player["player_identifier"],
                total_hits=player["total_hits"]
            )
            for player in chart_data["players"]
        ]
        
        return HitComparisonChartDto(
            chart_type=chart_data["chart_type"],
            players=players
        )
        
    except MatchNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": str(e)}
        )
    
    except DataUnavailableException as e:
        # UC-04 F1: Data not available
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "data_unavailable",
                "message": str(e),
                "reason": "Cannot generate chart without hit data"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "An unexpected error occurred", "details": str(e)}
        )


@router.get(
    "/{match_id}/players/{player_identifier}/hits",
    response_model=int,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": MatchErrorResponse, "description": "Match or player not found"},
        503: {"model": DataUnavailableResponse, "description": "Hit data not available"}
    },
    summary="Get hit count for specific player",
    description="""
    Get total hit count for a specific player in a match.
    
    Useful for detailed player analysis or comparison.
    """
)
async def get_player_hit_count(
    match_id: int,
    player_identifier: str,
    current_user: Player = Depends(get_current_user),
    match_service: MatchService = Depends(get_match_service)
) -> int:
    """
    Get hit count for a specific player
    
    Path parameters:
    - match_id: ID of the match
    - player_identifier: Player identifier (e.g., "player_1", "player_2")
    """
    try:
        hit_count = await match_service.get_player_hit_count(match_id, player_identifier)
        return hit_count
        
    except MatchNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": str(e)}
        )
    
    except PlayerInMatchNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": str(e)}
        )
    
    except DataUnavailableException as e:
        # UC-04 F1: Data not available
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "data_unavailable",
                "message": str(e),
                "reason": "Hit identification may have failed during analysis"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "An unexpected error occurred", "details": str(e)}
        )