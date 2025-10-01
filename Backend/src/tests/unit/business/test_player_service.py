import pytest
from unittest.mock import AsyncMock
from datetime import datetime

from app.business.services.player_service import PlayerService
from app.business.exceptions import (
    PlayerAlreadyExistsException, 
    PlayerNotFoundException, 
    ValidationException
)
from app.domain.player import Player


class TestPlayerServiceCreate:
    """Test suite for PlayerService.create_player method (UC-09)"""
    
    @pytest.fixture
    def player_service(self, mock_player_repository):
        """Create PlayerService with mocked repository"""
        return PlayerService(mock_player_repository)
    
    # ==================== SUCCESS SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_create_player_success(self, player_service, mock_player_repository, sample_player):
        """
        UC-09 S1: Successful player creation
        GIVEN valid player data
        WHEN creating a player
        THEN player should be created successfully
        """
        # Arrange
        mock_player_repository.get_by_id.return_value = None
        mock_player_repository.get_by_email.return_value = None
        mock_player_repository.create.return_value = sample_player
        
        # Act
        result = await player_service.create_player(
            id="firebase-uid-123",
            name="John Doe",
            email="john@example.com",
            role="player"
        )
        
        # Assert
        assert result.id == "firebase-uid-123"
        assert result.name == "John Doe"
        assert result.email == "john@example.com"
        assert result.role == "player"
        
        # Verify repository interactions
        mock_player_repository.get_by_id.assert_called_once_with("firebase-uid-123")
        mock_player_repository.get_by_email.assert_called_once_with("john@example.com")
        mock_player_repository.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_player_trims_name(self, player_service, mock_player_repository):
        """
        GIVEN name with leading/trailing whitespace
        WHEN creating a player
        THEN name should be trimmed
        """
        # Arrange
        mock_player_repository.get_by_id.return_value = None
        mock_player_repository.get_by_email.return_value = None
        created_player = Player(
            id="uid", name="John", email="john@example.com", role="player"
        )
        mock_player_repository.create.return_value = created_player
        
        # Act
        await player_service.create_player(
            id="uid",
            name="  John  ",
            email="john@example.com"
        )
        
        # Assert - check what was passed to create
        call_args = mock_player_repository.create.call_args
        created_entity = call_args[0][0]  # First positional argument
        assert created_entity.name == "John"
    
    @pytest.mark.asyncio
    async def test_create_player_lowercases_email(self, player_service, mock_player_repository):
        """
        GIVEN email with uppercase letters
        WHEN creating a player
        THEN email should be lowercased
        """
        # Arrange
        mock_player_repository.get_by_id.return_value = None
        mock_player_repository.get_by_email.return_value = None
        created_player = Player(
            id="uid", name="John", email="john@example.com", role="player"
        )
        mock_player_repository.create.return_value = created_player
        
        # Act
        await player_service.create_player(
            id="uid",
            name="John",
            email="John@EXAMPLE.COM"
        )
        
        # Assert
        call_args = mock_player_repository.create.call_args
        created_entity = call_args[0][0]
        assert created_entity.email == "john@example.com"
    
    # ==================== FAILURE SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_create_player_empty_firebase_uid(self, player_service, mock_player_repository):
        """
        UC-09 Validation: Firebase UID cannot be empty
        GIVEN empty Firebase UID
        WHEN creating a player
        THEN should raise ValidationException
        """
        # Act & Assert
        with pytest.raises(ValidationException) as exc_info:
            await player_service.create_player(
                id="",
                name="John Doe",
                email="john@example.com"
            )
        
        assert "Firebase UID" in str(exc_info.value)
        mock_player_repository.create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_create_player_whitespace_firebase_uid(self, player_service, mock_player_repository):
        """
        GIVEN Firebase UID with only whitespace
        WHEN creating a player
        THEN should raise ValidationException
        """
        with pytest.raises(ValidationException) as exc_info:
            await player_service.create_player(
                id="   ",
                name="John Doe",
                email="john@example.com"
            )
        
        assert "Firebase UID" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_player_empty_name(self, player_service, mock_player_repository):
        """
        UC-09 F4: Name cannot be empty
        GIVEN empty name (after trimming)
        WHEN creating a player
        THEN should raise ValidationException
        """
        with pytest.raises(ValidationException) as exc_info:
            await player_service.create_player(
                id="firebase-uid-123",
                name="",
                email="john@example.com"
            )
        
        assert "Name" in str(exc_info.value)
        assert "empty" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_create_player_whitespace_only_name(self, player_service, mock_player_repository):
        """
        UC-09 F4: Name cannot be only whitespace
        GIVEN name with only whitespace
        WHEN creating a player
        THEN should raise ValidationException after trimming
        """
        with pytest.raises(ValidationException) as exc_info:
            await player_service.create_player(
                id="firebase-uid-123",
                name="   ",
                email="john@example.com"
            )
        
        assert "Name" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_player_name_too_long(self, player_service, mock_player_repository):
        """
        UC-09 F5: Name cannot exceed 100 characters
        GIVEN name longer than 100 characters
        WHEN creating a player
        THEN should raise ValidationException
        """
        long_name = "a" * 101
        
        with pytest.raises(ValidationException) as exc_info:
            await player_service.create_player(
                id="firebase-uid-123",
                name=long_name,
                email="john@example.com"
            )
        
        assert "100" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_player_name_exactly_100_chars_is_valid(self, player_service, mock_player_repository):
        """
        GIVEN name with exactly 100 characters
        WHEN creating a player
        THEN should succeed (boundary test)
        """
        # Arrange
        exactly_100_chars = "a" * 100
        mock_player_repository.get_by_id.return_value = None
        mock_player_repository.get_by_email.return_value = None
        created_player = Player(
            id="uid", name=exactly_100_chars, email="test@example.com", role="player"
        )
        mock_player_repository.create.return_value = created_player
        
        # Act
        result = await player_service.create_player(
            id="uid",
            name=exactly_100_chars,
            email="test@example.com"
        )
        
        # Assert
        assert len(result.name) == 100
    
    @pytest.mark.asyncio
    async def test_create_player_empty_email(self, player_service, mock_player_repository):
        """
        GIVEN empty email
        WHEN creating a player
        THEN should raise ValidationException
        """
        with pytest.raises(ValidationException) as exc_info:
            await player_service.create_player(
                id="firebase-uid-123",
                name="John Doe",
                email=""
            )
        
        assert "Email" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_create_player_already_exists_by_id(self, player_service, mock_player_repository, sample_player):
        """
        UC-09 F6: Player already exists in database (by ID)
        GIVEN a player with the same Firebase UID already exists
        WHEN creating a player
        THEN should raise PlayerAlreadyExistsException
        """
        # Arrange - simulate existing player
        mock_player_repository.get_by_id.return_value = sample_player
        
        # Act & Assert
        with pytest.raises(PlayerAlreadyExistsException) as exc_info:
            await player_service.create_player(
                id="firebase-uid-123",
                name="John Doe",
                email="john@example.com"
            )
        
        assert "already exists" in str(exc_info.value).lower()
        mock_player_repository.create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_create_player_already_exists_by_email(self, player_service, mock_player_repository, sample_player):
        """
        UC-09 F6: Player already exists in database (by email)
        GIVEN a player with the same email already exists
        WHEN creating a player
        THEN should raise PlayerAlreadyExistsException
        """
        # Arrange
        mock_player_repository.get_by_id.return_value = None
        mock_player_repository.get_by_email.return_value = sample_player
        
        # Act & Assert
        with pytest.raises(PlayerAlreadyExistsException) as exc_info:
            await player_service.create_player(
                id="different-uid",
                name="Different Name",
                email="john@example.com"  # Same email as existing
            )
        
        assert "email" in str(exc_info.value).lower()
        mock_player_repository.create.assert_not_called()


class TestPlayerServiceGetById:
    """Test suite for PlayerService.get_player_by_id method"""
    
    @pytest.fixture
    def player_service(self, mock_player_repository):
        return PlayerService(mock_player_repository)
    
    @pytest.mark.asyncio
    async def test_get_player_by_id_success(self, player_service, mock_player_repository, sample_player):
        """
        GIVEN a player exists with the given ID
        WHEN getting player by ID
        THEN should return the player
        """
        # Arrange
        mock_player_repository.get_by_id.return_value = sample_player
        
        # Act
        result = await player_service.get_player_by_id("firebase-uid-123")
        
        # Assert
        assert result.id == "firebase-uid-123"
        assert result.name == "John Doe"
        mock_player_repository.get_by_id.assert_called_once_with("firebase-uid-123")
    
    @pytest.mark.asyncio
    async def test_get_player_by_id_not_found(self, player_service, mock_player_repository):
        """
        GIVEN no player exists with the given ID
        WHEN getting player by ID
        THEN should raise PlayerNotFoundException
        """
        # Arrange
        mock_player_repository.get_by_id.return_value = None
        
        # Act & Assert
        with pytest.raises(PlayerNotFoundException) as exc_info:
            await player_service.get_player_by_id("non-existent-id")
        
        assert "not found" in str(exc_info.value).lower()


class TestPlayerServiceGetByEmail:
    """Test suite for PlayerService.get_player_by_email method"""
    
    @pytest.fixture
    def player_service(self, mock_player_repository):
        return PlayerService(mock_player_repository)
    
    @pytest.mark.asyncio
    async def test_get_player_by_email_success(self, player_service, mock_player_repository, sample_player):
        """
        GIVEN a player exists with the given email
        WHEN getting player by email
        THEN should return the player
        """
        # Arrange
        mock_player_repository.get_by_email.return_value = sample_player
        
        # Act
        result = await player_service.get_player_by_email("john@example.com")
        
        # Assert
        assert result.email == "john@example.com"
        mock_player_repository.get_by_email.assert_called_once_with("john@example.com")
    
    @pytest.mark.asyncio
    async def test_get_player_by_email_not_found(self, player_service, mock_player_repository):
        """
        GIVEN no player exists with the given email
        WHEN getting player by email
        THEN should raise PlayerNotFoundException
        """
        # Arrange
        mock_player_repository.get_by_email.return_value = None
        
        # Act & Assert
        with pytest.raises(PlayerNotFoundException):
            await player_service.get_player_by_email("nonexistent@example.com")