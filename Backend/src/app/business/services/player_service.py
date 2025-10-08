from .interfaces import IPlayerService
from ..exceptions import PlayerAlreadyExistsException, PlayerNotFoundException, ValidationException
from ...data.repositories.interfaces import IPlayerRepository
from ...domain.player import Player

class PlayerService(IPlayerService):
    """Business service for Player operations"""
    
    def __init__(self, player_repository: IPlayerRepository):
        self._player_repository = player_repository
    
    async def create_player(
        self, 
        id: str,  # Firebase UID
        name: str, 
        email: str, 
        role: str = "player"
    ) -> Player:
        """Create a new player with Firebase UID"""
        
        name = name.strip()
        email = email.strip().lower()

        # Validation
        if not id or len(id.strip()) == 0:
            raise ValidationException("Firebase UID cannot be empty")
        
        if not name:
            raise ValidationException("Name cannot be empty or whitespace")
        
        if len(name) > 100:
            raise ValidationException("Name must be at most 100 characters long")

        if not email:
            raise ValidationException("Email cannot be empty")
        
        # Check if player already exists
        existing_player_by_id = await self._player_repository.get_by_id(id)
        if existing_player_by_id:
            raise PlayerAlreadyExistsException(f"Player with ID {id} already exists")
            
        existing_player_by_email = await self._player_repository.get_by_email(email)
        if existing_player_by_email:
            raise PlayerAlreadyExistsException(f"Player with email {email} already exists")
        
        # Create domain entity
        player = Player(
            id=id,
            name=name.strip(),
            email=email.strip().lower(),
            role=role
        )
        
        # Save through repository
        return await self._player_repository.create(player)
    
    async def get_player_by_id(self, player_id: str) -> Player:
        """Get player by ID"""
        player = await self._player_repository.get_by_id(player_id)
        if not player:
            raise PlayerNotFoundException(f"Player with ID {player_id} not found")
        return player
    
    async def get_player_by_email(self, email: str) -> Player:
        """Get player by email"""
        player = await self._player_repository.get_by_email(email)
        if not player:
            raise PlayerNotFoundException(f"Player with email {email} not found")
        return player
    
    # async def get_all_players(self) -> List[Player]:
    #     """Get all players"""
    #     return await self._player_repository.get_all()
    
    # async def update_player(self, player: Player) -> Player:
    #     """Update existing player"""
    #     if not player.id:
    #         raise ValidationException("Player ID is required for update")
        
    #     # Check if player exists
    #     existing_player = await self._player_repository.get_by_id(player.id)
    #     if not existing_player:
    #         raise PlayerNotFoundException(f"Player with ID {player.id} not found")
        
    #     # Validation
    #     if not player.name or len(player.name.strip()) == 0:
    #         raise ValidationException("Player name cannot be empty")
        
    #     if not player.email or len(player.email.strip()) == 0:
    #         raise ValidationException("Email cannot be empty")
        
    #     # Check if another player with the same email exists
    #     existing_player_by_email = await self._player_repository.get_by_email(player.email)
    #     if existing_player_by_email and existing_player_by_email.id != player.id:
    #         raise PlayerAlreadyExistsException(f"Another player with email {player.email} already exists")
        
    #     return await self._player_repository.update(player)
    
    # async def delete_player(self, player_id: str) -> bool:
    #     """Delete player by ID"""
    #     success = await self._player_repository.delete(player_id)
    #     if not success:
    #         raise PlayerNotFoundException(f"Player with ID {player_id} not found")
    #     return success