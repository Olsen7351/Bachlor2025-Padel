from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from ...data.connection import get_db_session
from ...data.repositories.player_repository import PlayerRepository
from ...business.services.player_service import PlayerService
from ...business.exceptions import PlayerNotFoundException
from ...auth.dependencies import get_current_user, AuthenticatedUser
from ..dtos.player_dto import PlayerResponse

router = APIRouter(prefix="/players", tags=["players"])

async def get_player_service(session: AsyncSession = Depends(get_db_session)) -> PlayerService:
    """Dependency injection for PlayerService"""
    player_repository = PlayerRepository(session)
    return PlayerService(player_repository)

@router.get("/{player_id}", response_model=PlayerResponse)
async def get_player(
    player_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),  # Now requires auth
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
    current_user: AuthenticatedUser = Depends(get_current_user),  # Now requires auth
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


# For testing purposes, you can add an endpoint to get current user's profile
@router.get("/me/profile", response_model=PlayerResponse)
async def get_my_profile(
    current_user: AuthenticatedUser = Depends(get_current_user),
    player_service: PlayerService = Depends(get_player_service)
):
    """Get current user's profile"""
    try:
        player = await player_service.get_player_by_id(current_user.uid)
        
        return PlayerResponse(
            id=player.id,
            name=player.name,
            email=player.email,
            role=player.role,
            created_at=player.created_at,
            updated_at=player.updated_at
        )
    
    except PlayerNotFoundException as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to get profile: {str(e)}")