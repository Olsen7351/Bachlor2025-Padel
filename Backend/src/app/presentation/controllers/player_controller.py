from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from ...business.services.interfaces import IPlayerService
from ...business.exceptions import PlayerNotFoundException
from ..dtos.player_dto import PlayerResponse
from ...domain.player import Player

# DI container
from ...dependencies import get_player_service
from ...auth.dependencies import get_current_user


router = APIRouter(prefix="/players", tags=["players"])


@router.get("/{player_id}", response_model=PlayerResponse)
async def get_player(
    player_id: str,
    current_user: Player = Depends(get_current_user),
    player_service: IPlayerService = Depends(get_player_service)
):
    """
    Get player by ID (requires authentication)
    
    Args:
        player_id: The player's unique identifier
        current_user: Authenticated user (injected)
        player_service: Player service (injected)
        
    Returns:
        PlayerResponse with player details
        
    Raises:
        404: Player not found
        500: Internal server error
    """
    try:
        player = await player_service.get_player_by_id(player_id)
        
        return PlayerResponse(
            id=player.id,
            name=player.name,
            email=player.email,
            role=player.role,
            created_at=player.created_at,
            updated_at=player.updated_at
        )
    
    except PlayerNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get player: {str(e)}"
        )


@router.get("/", response_model=List[PlayerResponse])
async def list_players(
    current_user: Player = Depends(get_current_user),
    player_service: IPlayerService = Depends(get_player_service)
):
    """
    Get all players (requires authentication)
    
    Returns:
        List of PlayerResponse objects
        
    Raises:
        500: Internal server error
    """
    try:
        players = await player_service.get_all_players()
        
        return [
            PlayerResponse(
                id=player.id,
                name=player.name,
                email=player.email,
                role=player.role,
                created_at=player.created_at,
                updated_at=player.updated_at
            )
            for player in players
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get players: {str(e)}"
        )


@router.get("/me/profile", response_model=PlayerResponse)
async def get_my_profile(
    current_user: Player = Depends(get_current_user)
):
    """
    Get current user's profile
    
    Note: current_user is already a Player from the database,
    so we don't need to call the service again.
    
    Returns:
        PlayerResponse with current user's details
    """
    try:
        return PlayerResponse(
            id=current_user.id,
            name=current_user.name,
            email=current_user.email,
            role=current_user.role,
            created_at=current_user.created_at,
            updated_at=current_user.updated_at
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to get profile: {str(e)}"
        )