from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from datetime import datetime

from ...domain.video import Video, VideoStatus
from ..models.video_model import VideoModel
from .interfaces import IVideoRepository


class VideoRepository(IVideoRepository):
    """
    Repository implementation for Video entity
    Extends base repository with video-specific operations
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, VideoModel)
    
    def _to_domain(self, model: VideoModel) -> Video:
        """Convert SQLAlchemy model to domain entity"""
        if model is None:
            return None
        
        return Video(
            id=model.id,
            file_name=model.file_name,
            storage_path=model.storage_path,
            status=VideoStatus(model.status),
            upload_timestamp=model.upload_timestamp,
            video_length=model.video_length,
            is_deleted=model.is_deleted,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
    
    def _to_model(self, domain: Video) -> VideoModel:
        """Convert domain entity to SQLAlchemy model"""
        return VideoModel(
            id=domain.id,
            file_name=domain.file_name,
            storage_path=domain.storage_path,
            status=domain.status.value,
            upload_timestamp=domain.upload_timestamp,
            video_length=domain.video_length,
            is_deleted=domain.is_deleted,
            created_at=domain.created_at,
            updated_at=domain.updated_at
        )
    
    async def get_by_id(self, id: int) -> Optional[Video]:
        """Get video by ID, excluding soft-deleted videos"""
        stmt = select(VideoModel).where(
            VideoModel.id == id,
            VideoModel.is_deleted == False
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_domain(model) if model else None
    
    async def get_all(self) -> List[Video]:
        """Get all non-deleted videos"""
        stmt = (
            select(VideoModel)
            .where(VideoModel.is_deleted == False)
            .order_by(VideoModel.upload_timestamp.desc())
        )
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]
    
    async def create(self, entity: Video) -> Video:
        """Create a new video record in database"""
        # Create model from domain entity
        model = VideoModel(
            file_name=entity.file_name,
            storage_path=entity.storage_path,
            status=entity.status.value,
            upload_timestamp=entity.upload_timestamp or datetime.utcnow(),
            video_length=entity.video_length,
            is_deleted=entity.is_deleted,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        
        return self._to_domain(model)
    
    async def update(self, entity: Video) -> Video:
        """Update existing video"""
        stmt = (
            update(VideoModel)
            .where(VideoModel.id == entity.id)
            .values(
                file_name=entity.file_name,
                storage_path=entity.storage_path,
                status=entity.status.value,
                video_length=entity.video_length,
                is_deleted=entity.is_deleted,
                updated_at=datetime.now()
            )
            .returning(VideoModel)
        )
        
        result = await self.session.execute(stmt)
        await self.session.flush()
        updated_model = result.scalar_one()
        
        return self._to_domain(updated_model)
    
    async def delete(self, id: int) -> bool:
        """Hard delete video (use soft_delete instead for normal operations)"""
        stmt = select(VideoModel).where(VideoModel.id == id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model:
            await self.session.delete(model)
            await self.session.flush()
            return True
        return False
    
    # Video-specific methods
    
    async def update_status(
        self, 
        video_id: int, 
        status: VideoStatus, 
        error_message: Optional[str] = None
    ) -> Video:
        """Update video status - video-specific operation"""
        stmt = (
            update(VideoModel)
            .where(VideoModel.id == video_id)
            .values(
                status=status.value,
                updated_at=datetime.now()
            )
            .returning(VideoModel)
        )
        
        result = await self.session.execute(stmt)
        await self.session.flush()
        updated_model = result.scalar_one()
        
        return self._to_domain(updated_model)
    
    async def get_by_status(self, status: VideoStatus) -> List[Video]:
        """Get all videos with specific status - video-specific query"""
        stmt = (
            select(VideoModel)
            .where(
                VideoModel.status == status.value,
                VideoModel.is_deleted == False
            )
            .order_by(VideoModel.upload_timestamp.desc())
        )
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(model) for model in models]
    
    async def soft_delete(self, video_id: int) -> bool:
        """Soft delete video (set is_deleted=True) - video-specific operation"""
        stmt = (
            update(VideoModel)
            .where(VideoModel.id == video_id)
            .values(is_deleted=True, updated_at=datetime.now())
        )
        
        result = await self.session.execute(stmt)
        await self.session.flush()
        
        return result.rowcount > 0