from abc import ABC, abstractmethod
from typing import Optional
from ...domain.player import Player

class IPlayerService(ABC):
    """
    Interface for Player business logic
    """
    
    @abstractmethod
    async def create_player(self, id: str, name: str, email: str, role: str = "player") -> Player:
        """Create a new player with validation"""
        pass
    
    @abstractmethod
    async def get_player_by_id(self, player_id: str) -> Player:
        """Get player by ID"""
        pass
    
    @abstractmethod
    async def get_player_by_email(self, email: str) -> Player:
        """Get player by email"""
        pass
    