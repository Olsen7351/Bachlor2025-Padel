"""Hits repository implementation"""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from ...domain.match import Hits
from ..models.match_model import HitsModel
from .interfaces import IHitsRepository


class HitsRepository(IHitsRepository):
    """Repository for Hits entity - standalone table"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, HitsModel)
    
    def _to_domain(self, model: HitsModel) -> Optional[Hits]:
        """Convert SQLAlchemy model to domain entity"""
        if model is None:
            return None
        
        return Hits(
            id=model.id,
            overhead_hits=model.overhead_hits,
            lob=model.lob,
            serve=model.serve,
            backhand=model.backhand,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    def _to_model(self, domain: Hits) -> HitsModel:
        """Convert domain entity to SQLAlchemy model"""
        return HitsModel(
            id=domain.id,
            overhead_hits=domain.overhead_hits,
            lob=domain.lob,
            serve=domain.serve,
            backhand=domain.backhand,
            created_at=domain.created_at,
            updated_at=domain.updated_at
        )
    
    async def get_by_id(self, id: int) -> Optional[Hits]:
        """Get hits by ID"""
        stmt = select(HitsModel).where(HitsModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)
    
    async def get_all(self) -> List[Hits]:
        """Get all hits records"""
        stmt = select(HitsModel)
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]
    
    async def create(self, entity: Hits) -> Hits:
        """Create a new hits record"""
        model = HitsModel(
            overhead_hits=entity.overhead_hits,
            lob=entity.lob,
            serve=entity.serve,
            backhand=entity.backhand,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        
        return self._to_domain(model)
    
    async def update(self, entity: Hits) -> Hits:
        """Update existing hits record"""
        stmt = select(HitsModel).where(HitsModel.id == entity.id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            model.overhead_hits = entity.overhead_hits
            model.lob = entity.lob
            model.serve = entity.serve
            model.backhand = entity.backhand
            model.updated_at = datetime.now()
            await self.session.flush()
            await self.session.refresh(model)
            return self._to_domain(model)
        
        return None
    
    async def delete(self, id: int) -> bool:
        """Delete hits record"""
        stmt = select(HitsModel).where(HitsModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            await self.session.delete(model)
            await self.session.flush()
            return True
        return False
