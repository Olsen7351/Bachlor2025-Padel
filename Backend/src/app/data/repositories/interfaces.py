from abc import abstractmethod
from typing import Optional
from .base_repository import BaseRepository
from ...domain.player import Player
from ..models.player_model import PlayerModel

class IPlayerRepository(BaseRepository[Player, PlayerModel]):
    """
    Player-specific repository interface
    Extends BaseRepository with player-specific methods
    
    Follows Interface Segregation Principle:
    - Base methods in BaseRepository
    - Player-specific methods here
    """
    
    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[Player]:
        """Get player by email - player-specific query"""
        pass
    