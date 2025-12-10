from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from ...domain.match import Heatmap
from ..models.match_model import HeatmapModel
from .interfaces import IHeatmapRepository


class HeatmapRepository(IHeatmapRepository):
    """Repository for Heatmap entity - stores binary PNG image data"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, HeatmapModel)
    
    def _to_domain(self, model: HeatmapModel) -> Optional[Heatmap]:
        """Convert SQLAlchemy model to domain entity"""
        if model is None:
            return None
        
        return Heatmap(
            id=model.id,
            heatmap=model.heatmap,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    def _to_model(self, domain: Heatmap) -> HeatmapModel:
        """Convert domain entity to SQLAlchemy model"""
        return HeatmapModel(
            id=domain.id,
            heatmap=domain.heatmap,
            created_at=domain.created_at,
            updated_at=domain.updated_at
        )
    
    async def get_by_id(self, id: int) -> Optional[Heatmap]:
        """Get heatmap by ID"""
        stmt = select(HeatmapModel).where(HeatmapModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model)
    
    async def get_all(self) -> List[Heatmap]:
        """Get all heatmap records"""
        stmt = select(HeatmapModel)
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]
    
    async def create(self, entity: Heatmap) -> Heatmap:
        """Create a new heatmap record"""
        model = HeatmapModel(
            heatmap=entity.heatmap,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        
        return self._to_domain(model)
    
    async def update(self, entity: Heatmap) -> Heatmap:
        """Update existing heatmap record"""
        stmt = select(HeatmapModel).where(HeatmapModel.id == entity.id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            model.heatmap = entity.heatmap
            model.updated_at = datetime.now()
            await self.session.flush()
            await self.session.refresh(model)
            return self._to_domain(model)
        
        return None
    
    async def delete(self, id: int) -> bool:
        """Delete heatmap record"""
        stmt = select(HeatmapModel).where(HeatmapModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            await self.session.delete(model)
            await self.session.flush()
            return True
        return False
