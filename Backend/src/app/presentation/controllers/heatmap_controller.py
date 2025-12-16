from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from typing import Annotated, Optional

# Business layer
from ...business.services.interfaces import IHeatmapService
from ...business.exceptions import (
    MatchNotFoundException,
    PlayerInMatchNotFoundException,
    DataUnavailableException,
    HeatmapNotFoundException,
    InsufficientPositionDataException
)
from ..dtos.heatmap_dto import (
    PlayerHeatmapDto,
    HeatmapComparisonDto,
    HeatmapDataDto,
    AvailableHeatmapsDto,
    HeatmapErrorResponse,
    HeatmapUnavailableResponse
)

from ...domain.player import Player

# DI container
from ...dependencies import get_heatmap_service

from ...auth.dependencies import get_current_user



router = APIRouter(prefix="/heatmaps", tags=["heatmaps"])


@router.get(
    "/matches/{match_id}/players/{player_identifier}",
    response_model=PlayerHeatmapDto,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": HeatmapErrorResponse, "description": "Match or player not found"},
        503: {"model": HeatmapUnavailableResponse, "description": "Heatmap data not available"}
    },
    summary="Get player heatmap",
    description="""
    Get heatmap visualization for a specific player in a match.
    Implements UC-02 Success Scenario S1.
    
    **Success Scenario S1:**
    - Returns 2D court map with heatmap overlay
    - Color intensity indicates time spent in each area
    - Data returned as base64 encoded PNG
    
    **Usage in frontend:**
    ```html
    <img src={`data:${response.content_type};base64,${response.heatmap_2d}`} />
    ```
    
    **Failure Scenarios:**
    - F1: Analysis data not available (503)
    - F2: Insufficient position data for this player (503)
    
    **Note:** Only player_1 and player_2 (near-side players) have heatmap data.
    """
)
async def get_player_heatmap(
    match_id: int,
    player_identifier: str,
    current_user: Player = Depends(get_current_user),
    heatmap_service: IHeatmapService = Depends(get_heatmap_service)
) -> PlayerHeatmapDto:
    """
    UC-02 S1: Get heatmap for a specific player.
    
    Path parameters:
    - match_id: ID of the match
    - player_identifier: Player identifier (e.g., "player_1", "player_2")
    """
    try:
        result = await heatmap_service.get_player_heatmap(match_id, player_identifier)
        
        return PlayerHeatmapDto(
            match_id=result.match_id,
            player_identifier=result.player_identifier,
            heatmap_2d=result.heatmap_2d,
            content_type=result.content_type
        )
        
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
    
    except InsufficientPositionDataException as e:
        # UC-02 F2: Insufficient tracking data
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "insufficient_data",
                "message": str(e),
                "reason": "The AI could not track this player sufficiently to generate a heatmap"
            }
        )
    
    except (DataUnavailableException, HeatmapNotFoundException) as e:
        # UC-02 F1: Data not available
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "heatmap_unavailable",
                "message": str(e),
                "reason": "Analysis may have failed or is incomplete"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "An unexpected error occurred", "details": str(e)}
        )


@router.get(
    "/matches/{match_id}/players/{player_identifier}/image",
    response_class=Response,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"content": {"image/png": {}}, "description": "Raw PNG image"},
        404: {"model": HeatmapErrorResponse},
        503: {"model": HeatmapUnavailableResponse}
    },
    summary="Get player heatmap as raw image",
    description="""
    Get heatmap as raw PNG image bytes.
    Useful for direct image URLs in img tags.
    
    **Usage:**
    ```html
    <img src="/api/heatmaps/matches/1/players/player_1/image" />
    ```
    """
)
async def get_player_heatmap_image(
    match_id: int,
    player_identifier: str,
    current_user: Player = Depends(get_current_user),
    heatmap_service: IHeatmapService = Depends(get_heatmap_service)
) -> Response:
    """
    Get raw heatmap image bytes.
    Returns PNG with appropriate content-type header.
    """
    try:
        image_bytes = await heatmap_service.get_player_heatmap_raw(
            match_id, player_identifier
        )
        
        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "Content-Disposition": f"inline; filename=heatmap_{player_identifier}.png"
            }
        )
        
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
    
    except (DataUnavailableException, HeatmapNotFoundException) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "heatmap_unavailable",
                "message": str(e),
                "reason": "Heatmap data not available"
            }
        )


@router.get(
    "/matches/{match_id}/compare",
    response_model=HeatmapComparisonDto,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": HeatmapErrorResponse},
        503: {"model": HeatmapUnavailableResponse}
    },
    summary="Compare multiple player heatmaps",
    description="""
    Get heatmaps for multiple players for comparison.
    Implements UC-02 Success Scenario S2.
    
    **Success Scenario S2:**
    - Returns heatmaps for all requested players
    - If no players specified, returns all available heatmaps
    - Allows visual comparison between players
    
    **Query parameters:**
    - players: Optional comma-separated list of player identifiers
              Example: ?players=player_1,player_2
    """
)
async def get_heatmap_comparison(
    match_id: int,
    players: Annotated[
        Optional[str], 
        Query(description="Comma-separated player identifiers (e.g., 'player_1,player_2')")
    ] = None,
    current_user: Player = Depends(get_current_user),
    heatmap_service: IHeatmapService = Depends(get_heatmap_service)
) -> HeatmapComparisonDto:
    """
    UC-02 S2: Get multiple heatmaps for comparison.
    """
    try:
        # Parse player identifiers
        player_identifiers = None
        if players:
            player_identifiers = [p.strip() for p in players.split(",")]
        
        result = await heatmap_service.get_heatmap_comparison(
            match_id, player_identifiers
        )
        
        return HeatmapComparisonDto(
            match_id=result.match_id,
            heatmaps=[
                HeatmapDataDto(
                    player_identifier=h.player_identifier,
                    heatmap_base64=h.heatmap_base64,
                    content_type=h.content_type
                )
                for h in result.heatmaps
            ]
        )
        
    except MatchNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": str(e)}
        )
    
    except DataUnavailableException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "heatmap_unavailable",
                "message": str(e),
                "reason": "No heatmap data available for comparison"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "An unexpected error occurred", "details": str(e)}
        )


@router.get(
    "/matches/{match_id}/available",
    response_model=AvailableHeatmapsDto,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": HeatmapErrorResponse}
    },
    summary="Get available heatmap players",
    description="""
    Get list of players who have heatmap data available.
    Useful for frontend to know which player tabs to enable.
    
    **Note:** Typically only player_1 and player_2 will have heatmaps,
    as they are the players on the analyzed side of the court.
    """
)
async def get_available_heatmap_players(
    match_id: int,
    current_user: Player = Depends(get_current_user),
    heatmap_service: IHeatmapService = Depends(get_heatmap_service)
) -> AvailableHeatmapsDto:
    """
    Get list of players with available heatmap data.
    """
    try:
        available = await heatmap_service.get_available_heatmap_players(match_id)
        
        return AvailableHeatmapsDto(
            match_id=match_id,
            available_players=available
        )
        
    except MatchNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": str(e)}
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "An unexpected error occurred", "details": str(e)}
        )