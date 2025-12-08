from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from ...data.connection import get_db_session
from ...data.repositories.player_repository import PlayerRepository
from ...business.services.player_service import PlayerService
from ...business.exceptions import PlayerNotFoundException
from ...auth.dependencies import get_current_user
from ...domain.player import Player
from ..dtos.player_dto import PlayerResponse

router = APIRouter(prefix="/players", tags=["players"])

async def get_player_service(session: AsyncSession = Depends(get_db_session)) -> PlayerService:
    """Dependency injection for PlayerService"""
    player_repository = PlayerRepository(session)
    return PlayerService(player_repository)

@router.get("/{player_id}", response_model=PlayerResponse)
async def get_player(
    player_id: str,
    current_user: Player = Depends(get_current_user),  # Returns Player, not AuthenticatedUser
    player_service: PlayerService = Depends(get_player_service)
):
    """Get player by ID (requires authentication)"""
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get player: {str(e)}")

@router.get("/", response_model=List[PlayerResponse])
async def list_players(
    current_user: Player = Depends(get_current_user),  # Returns Player
    player_service: PlayerService = Depends(get_player_service)
):
    """Get all players (requires authentication)"""
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get players: {str(e)}")


@router.get("/me/profile", response_model=PlayerResponse)
async def get_my_profile(
    current_user: Player = Depends(get_current_user),  # Returns Player with .id
    player_service: PlayerService = Depends(get_player_service)
):
    """Get current user's profile"""
    try:
        # current_user is already a Player from database
        # Just return it directly, no need to look up again
        return PlayerResponse(
            id=current_user.id,  # Use .id, not .uid
            name=current_user.name,
            email=current_user.email,
            role=current_user.role,
            created_at=current_user.created_at,
            updated_at=current_user.updated_at
        )
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get profile: {str(e)}")