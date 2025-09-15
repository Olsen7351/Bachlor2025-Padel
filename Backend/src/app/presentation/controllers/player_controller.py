from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from ...data.connection import get_db_session
from ...data.repositories.player_repository import PlayerRepository
from ...business.services.player_service import PlayerService
from ...business.exceptions import PlayerAlreadyExistsException, PlayerNotFoundException, ValidationException
from ..dtos.player_dto import PlayerCreateRequest, PlayerResponse

router = APIRouter(prefix="/players", tags=["players"])

async def get_player_service(session: AsyncSession = Depends(get_db_session)) -> PlayerService:
    """Dependency injection for PlayerService"""
    player_repository = PlayerRepository(session)
    return PlayerService(player_repository)

@router.post("/", response_model=PlayerResponse, status_code=status.HTTP_201_CREATED)
async def create_player(
    request: PlayerCreateRequest,
    player_service: PlayerService = Depends(get_player_service)
):
    """Create a new player"""
    try:
        player = await player_service.create_player(
            name=request.name,
            email=request.email,
            password=request.password,
            role=request.role or "player"
        )
        
        return PlayerResponse(
            id=player.id,
            name=player.name,
            email=player.email,
            role=player.role,
            created_at=player.created_at,
            updated_at=player.updated_at
        )
    
    except PlayerAlreadyExistsException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValidationException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # ADD DEBUG INFO
        print(f"DEBUG ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")

@router.get("/{player_id}", response_model=PlayerResponse)
async def get_player(
    player_id: int,
    player_service: PlayerService = Depends(get_player_service)
):
    """Get player by ID"""
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
        # ADD DEBUG INFO
        print(f"DEBUG ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")

@router.get("/", response_model=List[PlayerResponse])
async def list_players(
    player_service: PlayerService = Depends(get_player_service)
):
    """Get all players"""
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
        # ADD DEBUG INFO
        print(f"DEBUG ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")
