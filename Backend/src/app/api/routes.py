from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.connection import get_db_session
from app.models.database import Player
from app.models.schemas import PlayerCreate, PlayerResponse
from app.config import get_settings
from datetime import datetime

router = APIRouter()
settings = get_settings()


@router.get("/")
async def hello_world():
    return {
        "message": "Padel Analyzer API is running with database!",
        "version": settings.api_version,
        "environment": settings.environment,
    }


@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}


# Player endpoints with database
@router.post("/players/", response_model=PlayerResponse)
async def create_player(
    player_data: PlayerCreate, session: AsyncSession = Depends(get_db_session)
):
    # Check if email already exists
    stmt = select(Player).where(Player.email == player_data.email)
    existing_player = await session.scalar(stmt)

    if existing_player:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Create new player (TODO: Hash password in production!)
    new_player = Player(
        name=player_data.name,
        email=player_data.email,
        password=player_data.password,  # TODO: Hash this!
    )

    session.add(new_player)
    await session.commit()
    await session.refresh(new_player)

    return PlayerResponse.model_validate(new_player)


@router.get("/players/{player_id}", response_model=PlayerResponse)
async def get_player(player_id: int, session: AsyncSession = Depends(get_db_session)):
    player = await session.get(Player, player_id)
    if not player:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Player not found"
        )

    return PlayerResponse.model_validate(player)


@router.get("/players/", response_model=list[PlayerResponse])
async def list_players(session: AsyncSession = Depends(get_db_session)):
    stmt = select(Player)
    result = await session.execute(stmt)
    players = result.scalars().all()

    return [PlayerResponse.model_validate(player) for player in players]


# Test endpoint to check database connectivity
@router.get("/test-db")
async def test_database(session: AsyncSession = Depends(get_db_session)):
    try:
        # Simple query to test database connection
        result = await session.execute(select(1))
        value = result.scalar()

        # Count players
        stmt = select(Player)
        players_result = await session.execute(stmt)
        player_count = len(players_result.scalars().all())

        return {
            "database_connected": True,
            "test_query_result": value,
            "player_count": player_count,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection failed: {str(e)}",
        )
