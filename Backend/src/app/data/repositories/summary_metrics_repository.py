from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from ...domain.match import SummaryMetrics
from ..models.match_model import SummaryMetricsModel, MatchPlayerModel
from .interfaces import ISummaryMetricsRepository


class SummaryMetricsRepository(ISummaryMetricsRepository):
    """Repository for SummaryMetrics - works with SummaryMetricsModel"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, SummaryMetricsModel)
    
    def _to_domain(self, model: SummaryMetricsModel) -> Optional[SummaryMetrics]:
        """Convert SQLAlchemy model to domain entity"""
        if model is None:
            return None
        
        return SummaryMetrics(
            id=model.id,
            match_player_id=model.match_player_id,
            total_hits=model.total_hits,
            total_rallies=model.total_rallies,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    def _to_model(self, domain: SummaryMetrics) -> SummaryMetricsModel:
        """Convert domain entity to SQLAlchemy model"""
        return SummaryMetricsModel(
            id=domain.id,
            match_player_id=domain.match_player_id,
            total_hits=domain.total_hits,
            total_rallies=domain.total_rallies,
            created_at=domain.created_at,
            updated_at=domain.updated_at
        )
    
    async def get_by_id(self, id: int) -> Optional[SummaryMetrics]:
        """Get summary metrics by ID"""
        stmt = select(SummaryMetricsModel).where(SummaryMetricsModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)
    
    async def get_all(self) -> List[SummaryMetrics]:
        """Get all summary metrics"""
        stmt = select(SummaryMetricsModel).order_by(SummaryMetricsModel.created_at.desc())
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]
    
    async def create(self, entity: SummaryMetrics) -> SummaryMetrics:
        """Create new summary metrics record"""
        model = SummaryMetricsModel(
            match_player_id=entity.match_player_id,
            total_hits=entity.total_hits,
            total_rallies=entity.total_rallies,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        
        return self._to_domain(model)
    
    async def update(self, entity: SummaryMetrics) -> SummaryMetrics:
        """Update existing summary metrics"""
        stmt = select(SummaryMetricsModel).where(SummaryMetricsModel.id == entity.id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            model.total_hits = entity.total_hits
            model.total_rallies = entity.total_rallies
            model.updated_at = datetime.now()
            await self.session.flush()
            await self.session.refresh(model)
            return self._to_domain(model)
        
        return None
    
    async def delete(self, id: int) -> bool:
        """Delete summary metrics"""
        stmt = select(SummaryMetricsModel).where(SummaryMetricsModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            await self.session.delete(model)
            await self.session.flush()
            return True
        return False
    
    # SummaryMetrics-specific methods for UC-04
    
    async def get_by_match_player_id(self, match_player_id: int) -> Optional[SummaryMetrics]:
        """Get summary metrics for a specific match player"""
        stmt = select(SummaryMetricsModel).where(
            SummaryMetricsModel.match_player_id == match_player_id
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)
    
    async def get_all_by_match_id(self, match_id: int) -> List[SummaryMetrics]:
        """
        Get summary metrics for all players in a match
        UC-04 S1: Display all player hit counts
        
        Business Rule: Returns metrics ordered by hit count descending
        """
        stmt = (
            select(SummaryMetricsModel)
            .join(MatchPlayerModel, SummaryMetricsModel.match_player_id == MatchPlayerModel.id)
            .where(MatchPlayerModel.match_id == match_id)
            .order_by(SummaryMetricsModel.total_hits.desc())
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]
    
    async def get_by_match_and_set(self, match_id: int, set_number: int) -> List[SummaryMetrics]:
        """
        Get summary metrics filtered by set number
        UC-04 S2: Filter by set (placeholder for future implementation)
        
        Note: Set-level filtering requires Set entity implementation.
        Currently returns all metrics for the match.
        """
        # TODO: Implement set-level filtering when Set entity is added
        # This will require joining with Set table and filtering by set_number
        return await self.get_all_by_match_id(match_id)