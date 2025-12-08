from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, List, Union
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar('T')
TModel = TypeVar('TModel')

class BaseRepository(ABC, Generic[T, TModel]):
    """Base repository interface following Repository pattern"""
    
    def __init__(self, session: AsyncSession, model_class: type[TModel]):
        self.session = session
        self.model_class = model_class
    
    @abstractmethod
    def _to_domain(self, model: TModel) -> T:
        """Convert SQLAlchemy model to domain entity"""
        pass
    
    @abstractmethod
    def _to_model(self, domain: T) -> TModel:
        """Convert domain entity to SQLAlchemy model"""
        pass

    @abstractmethod
    async def get_by_id(self, id: Union[int, str]) -> Optional[T]:
        """Get entity by ID - supports both int and string IDs"""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[T]:
        """Get all entities"""
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create new entity"""
        pass
    
    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update existing entity"""
        pass
    
    @abstractmethod
    async def delete(self, id: Union[int, str]) -> bool:
        """Delete entity by ID - supports both int and string IDs"""
        pass