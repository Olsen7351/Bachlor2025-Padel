from fastapi import APIRouter, Depends, HTTPException, status

# Business layer
from ...business.services.interfaces import IMatchService
from ...business.exceptions import (
    MatchNotFoundException,
    PlayerInMatchNotFoundException,
    DataUnavailableException
)
from ..dtos.match_dto import (
    MatchSummaryDto,
    PlayerHitCountDto,
    DataUnavailableResponse,
    MatchErrorResponse
)

from ...domain.player import Player

# DI container
from ...dependencies import get_match_service

from ...auth.dependencies import get_current_user


router = APIRouter(prefix="/matches", tags=["matches"])


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
    match_service: IMatchService = Depends(get_match_service)
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
                total_hits=stat["total_hits"],
                overhead_hits=stat["overhead_hits"],
                lob=stat["lob"],
                serve=stat["serve"],
                groundstrokes=stat["groundstrokes"]
            )
            for stat in overview["player_statistics"]
        ]
        
        return MatchSummaryDto(
            match_id=overview["match_id"],
            analysis_id=overview["analysis_id"],
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
    match_service: IMatchService = Depends(get_match_service)
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