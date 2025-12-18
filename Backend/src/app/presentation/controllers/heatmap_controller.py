from fastapi import APIRouter, Depends, HTTPException, status, Response

# Business layer
from ...business.services.interfaces import IHeatmapService
from ...business.exceptions import (
    MatchNotFoundException,
    PlayerInMatchNotFoundException,
    DataUnavailableException,
    HeatmapNotFoundException
)
from ..dtos.heatmap_dto import (
    HeatmapErrorResponse,
    HeatmapUnavailableResponse
)

from ...domain.player import Player

# DI container
from ...dependencies import get_heatmap_service

from ...auth.dependencies import get_current_user



router = APIRouter(prefix="/heatmaps", tags=["heatmaps"])


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
    UC-02 S1: Get heatmap for a specific player.
    
    Path parameters:
    - match_id: ID of the match
    - player_identifier: Player identifier (e.g., "player_1", "player_2")

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