from typing import List, Optional
from ...domain.player import Player
from ...data.repositories.player_repository import PlayerRepository
from ..exceptions import PlayerAlreadyExistsException, PlayerNotFoundException, ValidationException
from datetime import datetime

class PlayerService:
    """Business logic for Player operations"""
    
    def __init__(self, player_repository: PlayerRepository):
        self.player_repository = player_repository
    
    async def create_player(self, name: str, email: str, password: str, role: str = "player") -> Player:
        """Create a new player with business validation"""
        
        # Simple validation
        self._validate_player_data(name, email, password)
        
        # Check if player already exists
        existing_player = await self.player_repository.get_by_email(email.lower().strip())
        if existing_player:
            raise PlayerAlreadyExistsException(f"Player with email {email} already exists")
        
        # Create domain entity (NO PASSWORD HASHING)
        player = Player(
            id=None,
            name=name.strip(),
            email=email.lower().strip(),
            password=password,  # Store as-is for now
            role=role,
            created_at=datetime.utcnow()
        )
        
        # Persist through repository
        return await self.player_repository.create(player)
    
    async def get_player_by_id(self, player_id: int) -> Player:
        """Get player by ID"""
        player = await self.player_repository.get_by_id(player_id)
        if not player:
            raise PlayerNotFoundException(f"Player with id {player_id} not found")
        return player
    
    async def get_all_players(self) -> List[Player]:
        """Get all players"""
        return await self.player_repository.get_all()
    
    def _validate_player_data(self, name: str, email: str, password: str) -> None:
        """Simple validation"""
        if not name or not name.strip():
            raise ValidationException("Name cannot be empty")
        if len(name.strip()) > 100:
            raise ValidationException("Name cannot be longer than 100 characters")
        
        if not email or not email.strip():
            raise ValidationException("Email cannot be empty")
        if "@" not in email or "." not in email.split("@")[-1]:
            raise ValidationException("Invalid email format")
        
        if not password:
            raise ValidationException("Password cannot be empty")
        if len(password) < 6:
            raise ValidationException("Password must be at least 6 characters long")