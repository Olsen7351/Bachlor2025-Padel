from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from datetime import datetime

from ...domain.analysis import Analysis
from ..models.analysis_model import AnalysisModel
from .interfaces import IAnalysisRepository


class AnalysisRepository(IAnalysisRepository):
    """
    Repository implementation for Analysis entity
    
    Responsibilities:
    - CRUD operations for Analysis entity
    - Analysis-specific queries
    
    Follows:
    - Single Responsibility Principle: Only handles Analysis data access
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, AnalysisModel)
    
    def _to_domain(self, model: AnalysisModel) -> Optional[Analysis]:
        """Convert SQLAlchemy model to domain entity"""
        if model is None:
            return None
        
        return Analysis(
            id=model.id,
            player_id=model.player_id,
            video_id=model.video_id,
            match_id=model.match_id,
            analysis_timestamp=model.analysis_timestamp,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    def _to_model(self, domain: Analysis) -> AnalysisModel:
        """Convert domain entity to SQLAlchemy model"""
        return AnalysisModel(
            id=domain.id,
            player_id=domain.player_id,
            video_id=domain.video_id,
            match_id=domain.match_id,
            analysis_timestamp=domain.analysis_timestamp,
            created_at=domain.created_at,
            updated_at=domain.updated_at
        )
    
    async def get_by_id(self, id: int) -> Optional[Analysis]:
        """Get analysis by ID"""
        stmt = select(AnalysisModel).where(AnalysisModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)
    
    async def get_all(self) -> List[Analysis]:
        """Get all analyses"""
        stmt = select(AnalysisModel).order_by(AnalysisModel.created_at.desc())
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]
    
    async def create(self, entity: Analysis) -> Analysis:
        """Create a new analysis record"""
        model = AnalysisModel(
            player_id=entity.player_id,
            video_id=entity.video_id,
            match_id=entity.match_id,
            analysis_timestamp=entity.analysis_timestamp or datetime.now(),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        
        return self._to_domain(model)
    
    async def update(self, entity: Analysis) -> Analysis:
        """Update existing analysis"""
        stmt = (
            update(AnalysisModel)
            .where(AnalysisModel.id == entity.id)
            .values(
                match_id=entity.match_id,
                analysis_timestamp=entity.analysis_timestamp,
                updated_at=datetime.now()
            )
            .returning(AnalysisModel)
        )
        
        result = await self.session.execute(stmt)
        await self.session.flush()
        updated_model = result.scalar_one()
        
        return self._to_domain(updated_model)
    
    async def delete(self, id: int) -> bool:
        """Delete analysis"""
        stmt = select(AnalysisModel).where(AnalysisModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            await self.session.delete(model)
            await self.session.flush()
            return True
        return False
    
    # Analysis-specific methods
    
    async def get_by_video_id(self, video_id: int) -> Optional[Analysis]:
        """
        Get analysis by video ID
        
        Business Rule: Each video has one analysis (1:1 relationship)
        Used to find analysis from video context
        """
        stmt = select(AnalysisModel).where(AnalysisModel.video_id == video_id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)
    
    async def get_by_match_id(self, match_id: int) -> Optional[Analysis]:
        """
        Get analysis by match ID
        
        Business Rule: Each match has one analysis (1:1 relationship)
        Used by MatchService to retrieve analysis_id for match overview
        """
        stmt = select(AnalysisModel).where(AnalysisModel.match_id == match_id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)
    
    async def get_by_player_id(self, player_id: str) -> List[Analysis]:
        """
        Get all analyses for a player
        
        Used for player history and statistics
        """
        stmt = (
            select(AnalysisModel)
            .where(AnalysisModel.player_id == player_id)
            .order_by(AnalysisModel.created_at.desc())
        )
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]