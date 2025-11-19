from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from datetime import datetime

from ...domain.match import Match, MatchPlayer
from ..models.match_model import MatchModel, MatchPlayerModel
from .interfaces import IMatchRepository, IMatchPlayerRepository

# Import AnalysisModel for the join query
from ..models.analysis_model import AnalysisModel


class MatchRepository(IMatchRepository):
    """
    Repository for Match entity - works with MatchModel
    
    Responsibilities:
    - CRUD operations for Match entity
    - Match-specific queries
    
    Follows:
    - Single Responsibility Principle: Only handles Match data access
    - Dependency Inversion Principle: Implements IMatchRepository interface
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, MatchModel)
    
    def _to_domain(self, model: MatchModel) -> Optional[Match]:
        """Convert SQLAlchemy model to domain entity"""
        if model is None:
            return None
        
        return Match(
            id=model.id,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    def _to_model(self, domain: Match) -> MatchModel:
        """Convert domain entity to SQLAlchemy model"""
        return MatchModel(
            id=domain.id,
            created_at=domain.created_at,
            updated_at=domain.updated_at
        )
    
    async def get_by_id(self, id: int) -> Optional[Match]:
        """Get match by ID"""
        stmt = select(MatchModel).where(MatchModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)
    
    async def get_all(self) -> List[Match]:
        """Get all matches"""
        stmt = select(MatchModel).order_by(MatchModel.created_at.desc())
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]
    
    async def create(self, entity: Match) -> Match:
        """Create a new match record"""
        model = MatchModel(
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        
        return self._to_domain(model)
    
    async def update(self, entity: Match) -> Match:
        """Update existing match"""
        stmt = select(MatchModel).where(MatchModel.id == entity.id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            model.updated_at = datetime.now()
            await self.session.flush()
            await self.session.refresh(model)
            return self._to_domain(model)
        
        return None
    
    async def delete(self, id: int) -> bool:
        """Delete match"""
        stmt = select(MatchModel).where(MatchModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            await self.session.delete(model)
            await self.session.flush()
            return True
        return False
    
    # Match-specific methods
    
    async def get_by_analysis_id(self, analysis_id: int) -> Optional[Match]:
        """
        Get match by analysis ID
        
        Business Rule: Analysis has 1:1 relationship with Match
        Used to find match from analysis context
        """
        stmt = select(MatchModel).join(AnalysisModel).where(
            AnalysisModel.id == analysis_id,
            AnalysisModel.match_id == MatchModel.id
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)
    
    async def get_match_with_players(self, match_id: int) -> Optional[Match]:
        """
        Get match with all associated players using eager loading
        
        Performance optimization: Uses selectinload to avoid N+1 queries
        """
        stmt = (
            select(MatchModel)
            .options(selectinload(MatchModel.match_players))
            .where(MatchModel.id == match_id)
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)


class MatchPlayerRepository(IMatchPlayerRepository):
    """
    Repository for MatchPlayer entity - works with MatchPlayerModel
    
    Responsibilities:
    - CRUD operations for MatchPlayer entity
    - Player-match relationship queries
    
    Follows:
    - Single Responsibility Principle: Only handles MatchPlayer data access
    - Dependency Inversion Principle: Implements IMatchPlayerRepository interface
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, MatchPlayerModel)
    
    def _to_domain(self, model: MatchPlayerModel) -> Optional[MatchPlayer]:
        """Convert SQLAlchemy model to domain entity"""
        if model is None:
            return None
        
        return MatchPlayer(
            id=model.id,
            match_id=model.match_id,
            player_identifier=model.player_identifier,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    def _to_model(self, domain: MatchPlayer) -> MatchPlayerModel:
        """Convert domain entity to SQLAlchemy model"""
        return MatchPlayerModel(
            id=domain.id,
            match_id=domain.match_id,
            player_identifier=domain.player_identifier,
            created_at=domain.created_at,
            updated_at=domain.updated_at
        )
    
    async def get_by_id(self, id: int) -> Optional[MatchPlayer]:
        """Get match player by ID"""
        stmt = select(MatchPlayerModel).where(MatchPlayerModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)
    
    async def get_all(self) -> List[MatchPlayer]:
        """Get all match players"""
        stmt = select(MatchPlayerModel)
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]
    
    async def create(self, entity: MatchPlayer) -> MatchPlayer:
        """Create a new match player record"""
        model = MatchPlayerModel(
            match_id=entity.match_id,
            player_identifier=entity.player_identifier,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        
        return self._to_domain(model)
    
    async def update(self, entity: MatchPlayer) -> MatchPlayer:
        """Update existing match player"""
        stmt = select(MatchPlayerModel).where(MatchPlayerModel.id == entity.id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            model.player_identifier = entity.player_identifier
            model.updated_at = datetime.now()
            await self.session.flush()
            await self.session.refresh(model)
            return self._to_domain(model)
        
        return None
    
    async def delete(self, id: int) -> bool:
        """Delete match player"""
        stmt = select(MatchPlayerModel).where(MatchPlayerModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            await self.session.delete(model)
            await self.session.flush()
            return True
        return False
    
    # MatchPlayer-specific methods
    
    async def get_by_match_id(self, match_id: int) -> List[MatchPlayer]:
        """
        Get all players for a specific match
        
        Business Rule: A match has exactly 4 players (player_1 through player_4)
        """
        stmt = select(MatchPlayerModel).where(MatchPlayerModel.match_id == match_id)
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]
    
    async def get_by_identifier(self, match_id: int, player_identifier: str) -> Optional[MatchPlayer]:
        """
        Get a specific player by their identifier in a match
        
        Business Rule: Player identifier must be unique within a match
        Used for looking up specific player statistics
        """
        stmt = select(MatchPlayerModel).where(
            MatchPlayerModel.match_id == match_id,
            MatchPlayerModel.player_identifier == player_identifier
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)