from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from .base_repository import BaseRepository
from ..models.player_model import PlayerModel
from ...domain.player import Player

class PlayerRepository(BaseRepository[Player, PlayerModel]):
    """Repository for Player domain entities"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, PlayerModel)
    
    async def get_by_id(self, id: str) -> Optional[Player]:
        """Get player by ID"""
        model = await self.session.get(PlayerModel, id)
        return self._to_domain(model) if model else None
    
    async def get_by_email(self, email: str) -> Optional[Player]:
        """Get player by email"""
        stmt = select(PlayerModel).where(PlayerModel.email == email)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model) if model else None
    
    async def get_all(self) -> List[Player]:
        """Get all players"""
        stmt = select(PlayerModel)
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]
    
    async def create(self, player: Player) -> Player:
        """Create new player"""
        model = self._to_model(player)
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return self._to_domain(model)
    
    async def update(self, player: Player) -> Player:
        """Update existing player"""
        model = await self.session.get(PlayerModel, player.id)
        if not model:
            raise ValueError(f"Player with id {player.id} not found")
        
        # Update fields
        model.name = player.name
        model.email = player.email
        model.role = player.role
        
        await self.session.commit()
        await self.session.refresh(model)
        return self._to_domain(model)
    
    async def delete(self, id: str) -> bool:
        """Delete player by ID"""
        model = await self.session.get(PlayerModel, id)
        if not model:
            return False
        
        await self.session.delete(model)
        await self.session.commit()
        return True
    
    def _to_domain(self, model: PlayerModel) -> Player:
        """Convert SQLAlchemy model to domain entity"""
        return Player(
            id=model.id,
            name=model.name,
            email=model.email,
            role=model.role,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    def _to_model(self, domain: Player) -> PlayerModel:
        """Convert domain entity to SQLAlchemy model"""
        model = PlayerModel(
            name=domain.name,
            email=domain.email,
            role=domain.role
        )
        
        # Only set ID if it's provided (Firebase will provide this)
        if domain.id:
            model.id = domain.id
            
        return model