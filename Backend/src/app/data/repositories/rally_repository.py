from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from ...domain.match import Rally
from ..models.match_model import RallyModel
from .interfaces import IRallyRepository


class RallyRepository(IRallyRepository):
    """
    Repository for Rally entity
    
    Responsibilities:
    - CRUD operations for Rally entity
    - Rally-specific queries (by summary_metrics_id)
    
    Design Note:
    - Rallies are match-level data stored via SummaryMetrics
    - All players in a match share the same rally data
    - Rally duration is the primary stored metric
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, RallyModel)
    
    def _to_domain(self, model: RallyModel) -> Optional[Rally]:
        """Convert SQLAlchemy model to domain entity"""
        if model is None:
            return None
        
        return Rally(
            id=model.id,
            summary_metrics_id=model.summary_metrics_id,
            duration=model.duration,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    def _to_model(self, domain: Rally) -> RallyModel:
        """Convert domain entity to SQLAlchemy model"""
        return RallyModel(
            id=domain.id,
            summary_metrics_id=domain.summary_metrics_id,
            duration=domain.duration,
            created_at=domain.created_at,
            updated_at=domain.updated_at
        )
    
    async def get_by_id(self, id: int) -> Optional[Rally]:
        """Get rally by ID"""
        stmt = select(RallyModel).where(RallyModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)
    
    async def get_all(self) -> List[Rally]:
        """Get all rallies"""
        stmt = select(RallyModel).order_by(RallyModel.id)
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]
    
    async def create(self, entity: Rally) -> Rally:
        """Create a new rally record"""
        model = RallyModel(
            summary_metrics_id=entity.summary_metrics_id,
            duration=entity.duration,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        
        return self._to_domain(model)
    
    async def update(self, entity: Rally) -> Rally:
        """Update existing rally"""
        stmt = select(RallyModel).where(RallyModel.id == entity.id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            model.duration = entity.duration
            model.updated_at = datetime.now()
            await self.session.flush()
            await self.session.refresh(model)
            return self._to_domain(model)
        
        return None
    
    async def delete(self, id: int) -> bool:
        """Delete rally record"""
        stmt = select(RallyModel).where(RallyModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            await self.session.delete(model)
            await self.session.flush()
            return True
        return False
    
    # Rally-specific methods
    
    async def get_by_summary_metrics_id(self, summary_metrics_id: int) -> List[Rally]:
        """
        Get all rallies for a summary metrics record.
        
        Note: Rallies are stored under SummaryMetrics but represent
        match-level data. Both players in a match share the same rallies.
        """
        stmt = (
            select(RallyModel)
            .where(RallyModel.summary_metrics_id == summary_metrics_id)
            .order_by(RallyModel.id)
        )
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]