"""
Unit tests for auth controller endpoints

Tests UC-00: Player Login and UC-09: Player Registration
"""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from datetime import datetime

from app.presentation.controllers.auth_controller import login, get_current_user_info, register
from app.auth.dependencies import AuthenticatedUser
from app.business.exceptions import PlayerNotFoundException, PlayerAlreadyExistsException, ValidationException
from app.domain.player import Player


class TestLoginEndpoint:
    """Test suite for /auth/login endpoint (UC-00)"""
    
    @pytest.fixture
    def authenticated_user(self):
        """Mock authenticated user from Firebase token"""
        return AuthenticatedUser(
            uid="firebase-uid-123",
            email="john@example.com",
            firebase_data={
                'uid': 'firebase-uid-123',
                'email': 'john@example.com',
                'email_verified': True
            }
        )
    
    @pytest.fixture
    def mock_player_service(self):
        """Mock player service"""
        return AsyncMock()
    
    # ==================== SUCCESS SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_login_success(self, authenticated_user, mock_player_service, sample_player):
        """
        UC-00 S1: Successful login
        GIVEN valid Firebase token and user exists in database
        WHEN calling login endpoint
        THEN should return user data
        """
        # Arrange
        mock_player_service.get_player_by_id.return_value = sample_player
        
        # Act
        result = await login(
            firebase_user=authenticated_user,
            player_service=mock_player_service
        )
        
        # Assert
        assert result.message == "Login successful"
        assert result.user.id == "firebase-uid-123"
        assert result.user.name == "John Doe"
        assert result.user.email == "john@example.com"
        mock_player_service.get_player_by_id.assert_called_once_with("firebase-uid-123")
    
    # ==================== FAILURE SCENARIOS ====================
    
    @pytest.mark.asyncio
    async def test_login_user_not_found_in_database(self, authenticated_user, mock_player_service):
        """
        UC-00 F3: User exists in Firebase but not in backend database
        GIVEN valid Firebase token but user not in database
        WHEN calling login endpoint
        THEN should return 404 error
        """
        # Arrange
        mock_player_service.get_player_by_id.side_effect = PlayerNotFoundException(
            "Player not found"
        )
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await login(
                firebase_user=authenticated_user,
                player_service=mock_player_service
            )
        
        assert exc_info.value.status_code == 404
        assert "complete registration" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_login_service_error(self, authenticated_user, mock_player_service):
        """
        GIVEN service throws unexpected error
        WHEN calling login endpoint
        THEN should return 500 error
        """
        # Arrange
        mock_player_service.get_player_by_id.side_effect = Exception("Database connection failed")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await login(
                firebase_user=authenticated_user,
                player_service=mock_player_service
            )
        
        assert exc_info.value.status_code == 500
        assert "failed" in exc_info.value.detail.lower()


class TestGetCurrentUserEndpoint:
    """Test suite for /auth/me endpoint"""
    
    @pytest.fixture
    def authenticated_user(self):
        """Mock authenticated user"""
        return AuthenticatedUser(
            uid="firebase-uid-123",
            email="john@example.com",
            firebase_data={'uid': 'firebase-uid-123', 'email': 'john@example.com'}
        )
    
    @pytest.fixture
    def mock_player_service(self):
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_get_me_success(self, authenticated_user, mock_player_service, sample_player):
        """
        GIVEN valid authenticated user
        WHEN calling /me endpoint
        THEN should return user profile
        """
        # Arrange
        mock_player_service.get_player_by_id.return_value = sample_player
        
        # Act
        result = await get_current_user_info(
            firebase_user=authenticated_user,
            player_service=mock_player_service
        )
        
        # Assert
        assert result.id == "firebase-uid-123"
        assert result.name == "John Doe"
        assert result.email == "john@example.com"
        mock_player_service.get_player_by_id.assert_called_once_with("firebase-uid-123")
    
    @pytest.mark.asyncio
    async def test_get_me_user_not_found(self, authenticated_user, mock_player_service):
        """
        GIVEN valid token but user not in database
        WHEN calling /me endpoint
        THEN should return 404 error
        """
        # Arrange
        mock_player_service.get_player_by_id.side_effect = PlayerNotFoundException("Not found")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_info(
                firebase_user=authenticated_user,
                player_service=mock_player_service
            )
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_get_me_service_error(self, authenticated_user, mock_player_service):
        """
        GIVEN service error
        WHEN calling /me endpoint
        THEN should return 500 error
        """
        # Arrange
        mock_player_service.get_player_by_id.side_effect = Exception("Internal error")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_info(
                firebase_user=authenticated_user,
                player_service=mock_player_service
            )
        
        assert exc_info.value.status_code == 500


class TestRegisterEndpoint:
    """Test suite for /auth/register endpoint (UC-09)"""
    
    @pytest.fixture
    def authenticated_user(self):
        return AuthenticatedUser(
            uid="firebase-uid-new",
            email="newuser@example.com",
            firebase_data={'uid': 'firebase-uid-new', 'email': 'newuser@example.com'}
        )
    
    @pytest.fixture
    def mock_player_service(self):
        return AsyncMock()
    
    @pytest.fixture
    def register_request(self):
        """Mock registration request DTO"""
        from app.presentation.dtos.auth_dto import RegisterRequest
        return RegisterRequest(name="New User")
    
    @pytest.mark.asyncio
    async def test_register_success(self, authenticated_user, mock_player_service, register_request):
        """
        UC-09 S1: Successful registration
        GIVEN valid Firebase token and name
        WHEN registering
        THEN should create player in database
        """
        # Arrange
        created_player = Player(
            id="firebase-uid-new",
            name="New User",
            email="newuser@example.com",
            role="player",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        mock_player_service.create_player.return_value = created_player
        
        # Act
        result = await register(
            request=register_request,
            firebase_user=authenticated_user,
            player_service=mock_player_service
        )
        
        # Assert
        assert result.id == "firebase-uid-new"
        assert result.name == "New User"
        assert result.email == "newuser@example.com"
        assert result.role == "player"
        
        mock_player_service.create_player.assert_called_once_with(
            id="firebase-uid-new",
            name="New User",
            email="newuser@example.com",
            role="player"
        )
    
    @pytest.mark.asyncio
    async def test_register_player_already_exists(
        self, authenticated_user, mock_player_service, register_request
    ):
        """
        UC-09 F6: Player already exists
        GIVEN user already registered
        WHEN attempting to register again
        THEN should return 400 error
        """
        # Arrange
        mock_player_service.create_player.side_effect = PlayerAlreadyExistsException(
            "Player already exists"
        )
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await register(
                request=register_request,
                firebase_user=authenticated_user,
                player_service=mock_player_service
            )
        
        assert exc_info.value.status_code == 400
        assert "already exists" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_register_validation_error(
        self, authenticated_user, mock_player_service, register_request
    ):
        """
        UC-09 F4/F5: Validation errors
        GIVEN invalid player data
        WHEN attempting to register
        THEN should return 400 error with validation message
        """
        # Arrange
        mock_player_service.create_player.side_effect = ValidationException(
            "Name cannot be empty"
        )
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await register(
                request=register_request,
                firebase_user=authenticated_user,
                player_service=mock_player_service
            )
        
        assert exc_info.value.status_code == 400
        assert "Name cannot be empty" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_register_unexpected_error(
        self, authenticated_user, mock_player_service, register_request
    ):
        """
        GIVEN unexpected service error
        WHEN attempting to register
        THEN should return 500 error
        """
        # Arrange
        mock_player_service.create_player.side_effect = Exception("Database error")
        
        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await register(
                request=register_request,
                firebase_user=authenticated_user,
                player_service=mock_player_service
            )
        
        assert exc_info.value.status_code == 500
        assert "failed" in exc_info.value.detail.lower()